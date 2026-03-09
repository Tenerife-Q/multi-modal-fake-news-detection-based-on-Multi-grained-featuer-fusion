import argparse
import csv
import os
import re
from collections import Counter

import pandas as pd


def normalize_text(value):
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text.strip()


def extract_keywords(text, topk=6):
    text = normalize_text(text).lower()
    if not text:
        return []

    # 英文词 + 中文连续片段
    en_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", text)
    zh_tokens = re.findall(r"[\u4e00-\u9fff]{2,6}", text)

    stopwords = {
        "http", "https", "www", "com", "amp", "nan", "这是", "我们", "你们", "他们", "一个", "这个", "那个",
        "而且", "已经", "不是", "就是", "可以", "因为", "什么", "为什么", "真的", "还是", "进行", "表示",
    }

    all_tokens = []
    for token in en_tokens + zh_tokens:
        t = token.strip()
        if len(t) < 2:
            continue
        if t in stopwords:
            continue
        all_tokens.append(t)

    freq = Counter(all_tokens)
    return [w for w, _ in freq.most_common(topk)]


def load_weibo_rows(path):
    df = pd.read_csv(path)

    rows = []
    for i, row in df.iterrows():
        image = normalize_text(row.get("images", "")) or normalize_text(row.get("image", ""))
        label = row.get("label", None)
        text = normalize_text(row.get("content", ""))
        text += " " + normalize_text(row.get("contentc", ""))
        text += " " + normalize_text(row.get("contentp", ""))
        text += " " + normalize_text(row.get("text", ""))
        text += " " + normalize_text(row.get("clip", ""))
        text += " " + normalize_text(row.get("textocr", ""))

        if text.strip() == "":
            continue

        rows.append({
            "id": f"weibo_{i}",
            "image": image,
            "label": int(label) if pd.notna(label) else -1,
            "text": text,
        })
    return rows


def load_gossip_rows(path):
    # gossip csv 非常大，低内存读取
    df = pd.read_csv(path, low_memory=False)

    rows = []
    for i, row in df.iterrows():
        text = normalize_text(row.get("content", ""))
        title = normalize_text(row.get("title", ""))
        merged = (title + " " + text).strip()
        if merged == "":
            continue

        image = normalize_text(row.get("image", "")) or normalize_text(row.get("top_img", ""))
        label = row.get("label", None)
        sid = normalize_text(row.get("id", "")) or f"gossip_{i}"

        rows.append({
            "id": f"gossip_{sid}",
            "image": image,
            "label": int(label) if pd.notna(label) else -1,
            "text": merged,
        })
    return rows


def build_triples(rows, topk=6):
    triples = set()

    for row in rows:
        sid = row["id"]
        label = row["label"]
        image = row["image"]

        if label in [0, 1]:
            label_name = "rumor" if label == 0 else "nonrumor"
            triples.add((sid, "has_label", label_name))

        if image:
            triples.add((sid, "has_image", image))

        kws = extract_keywords(row["text"], topk=topk)
        for kw in kws:
            triples.add((sid, "mentions", kw))

        for i in range(len(kws)):
            for j in range(i + 1, len(kws)):
                h, t = kws[i], kws[j]
                triples.add((h, "cooccur", t))
                triples.add((t, "cooccur", h))

    return triples


def main():
    parser = argparse.ArgumentParser(description="Build triples only from Weibo + Gossip datasets")
    parser.add_argument("--weibo_train", type=str, default="train_weibov.csv")
    parser.add_argument("--weibo_test", type=str, default="test_weibov.csv")
    parser.add_argument("--gossip_train", type=str, default="gossip/train_gossipcop.csv")
    parser.add_argument("--gossip_test", type=str, default="gossip/test_gossipcop.csv")
    parser.add_argument("--topk", type=int, default=6, help="top-k keywords per sample")
    parser.add_argument("--out", type=str, default="data/kg/weibo_gossip_triples.tsv")
    args = parser.parse_args()

    all_rows = []

    for path in [args.weibo_train, args.weibo_test]:
        if os.path.exists(path):
            all_rows.extend(load_weibo_rows(path))

    for path in [args.gossip_train, args.gossip_test]:
        if os.path.exists(path):
            all_rows.extend(load_gossip_rows(path))

    if len(all_rows) == 0:
        raise RuntimeError("No rows loaded from weibo/gossip files.")

    triples = build_triples(all_rows, topk=args.topk)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for h, r, t in sorted(triples):
            writer.writerow([h, r, t])

    print(f"saved triples: {args.out}")
    print(f"rows: {len(all_rows)}")
    print(f"triples: {len(triples)}")


if __name__ == "__main__":
    main()
