import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import pandas as pd


def normalize_text(value):
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def extract_keywords(text, topk=8):
    text = normalize_text(text).lower()
    if not text:
        return []

    en_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", text)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,8}", text)

    stopwords = {
        "http", "https", "www", "com", "amp", "nan", "这是", "我们", "你们", "他们", "一个", "这个", "那个",
        "而且", "已经", "不是", "就是", "可以", "因为", "什么", "为什么", "真的", "还是", "进行", "表示",
        "the", "and", "for", "with", "that", "from", "this", "was", "are", "have", "has", "will", "you", "your",
    }

    tokens = []
    for token in en_tokens + zh_tokens:
        t = token.strip()
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        tokens.append(t)

    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(topk)]


def load_weibo_rows(path):
    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        text = " ".join([
            normalize_text(row.get("content", "")),
            normalize_text(row.get("contentc", "")),
            normalize_text(row.get("contentp", "")),
            normalize_text(row.get("text", "")),
            normalize_text(row.get("clip", "")),
            normalize_text(row.get("textocr", "")),
        ]).strip()
        if text:
            rows.append(text)
    return rows


def load_gossip_rows(path):
    df = pd.read_csv(path, low_memory=False)
    rows = []
    for _, row in df.iterrows():
        text = " ".join([
            normalize_text(row.get("title", "")),
            normalize_text(row.get("content", "")),
            normalize_text(row.get("text", "")),
        ]).strip()
        if text:
            rows.append(text)
    return rows


def build_clean_triples(text_rows, topk=8, min_cooccur=3, max_neighbors=30):
    pair_counter = Counter()
    entity_freq = Counter()

    for text in text_rows:
        kws = list(dict.fromkeys(extract_keywords(text, topk=topk)))
        for kw in kws:
            entity_freq[kw] += 1
        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                a, b = kws[i], kws[j]
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                pair_counter[(a, b)] += 1

    neighbor_map = defaultdict(list)
    for (a, b), cnt in pair_counter.items():
        if cnt < min_cooccur:
            continue
        neighbor_map[a].append((b, cnt))
        neighbor_map[b].append((a, cnt))

    triples = set()
    for src, neighbors in neighbor_map.items():
        neighbors = sorted(neighbors, key=lambda x: (-x[1], x[0]))[:max_neighbors]
        for dst, cnt in neighbors:
            triples.add((src, "cooccur", dst))
            triples.add((dst, "cooccur", src))

    # 高频实体类型三元组（可选，增强图结构）
    for ent, freq in entity_freq.items():
        if freq >= max(10, min_cooccur * 2):
            triples.add((ent, "entity_type", "frequent_term"))

    return triples, entity_freq, pair_counter


def main():
    parser = argparse.ArgumentParser(description="Build clean triples from Weibo + Gossip (entity-entity only)")
    parser.add_argument("--weibo_train", type=str, default="train_weibov.csv")
    parser.add_argument("--weibo_test", type=str, default="test_weibov.csv")
    parser.add_argument("--gossip_train", type=str, default="gossip/train_gossipcop.csv")
    parser.add_argument("--gossip_test", type=str, default="gossip/test_gossipcop.csv")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min_cooccur", type=int, default=3)
    parser.add_argument("--max_neighbors", type=int, default=30)
    parser.add_argument("--out", type=str, default="data/kg/weibo_gossip_triples_clean.tsv")
    args = parser.parse_args()

    text_rows = []
    for p in [args.weibo_train, args.weibo_test]:
        if os.path.exists(p):
            text_rows.extend(load_weibo_rows(p))
    for p in [args.gossip_train, args.gossip_test]:
        if os.path.exists(p):
            text_rows.extend(load_gossip_rows(p))

    if not text_rows:
        raise RuntimeError("No text rows loaded from weibo/gossip files.")

    triples, entity_freq, pair_counter = build_clean_triples(
        text_rows,
        topk=args.topk,
        min_cooccur=args.min_cooccur,
        max_neighbors=args.max_neighbors,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for h, r, t in sorted(triples):
            writer.writerow([h, r, t])

    print(f"saved clean triples: {args.out}")
    print(f"text_rows: {len(text_rows)}")
    print(f"unique_entities: {len(entity_freq)}")
    print(f"candidate_pairs: {len(pair_counter)}")
    print(f"triples: {len(triples)}")


if __name__ == "__main__":
    main()
