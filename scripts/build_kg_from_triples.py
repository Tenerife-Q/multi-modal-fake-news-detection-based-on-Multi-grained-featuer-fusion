import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
from transformers import BertTokenizer


def detect_delimiter(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return "\t"


def read_triples(path):
    delimiter = detect_delimiter(path)
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 3:
                continue
            h = str(row[0]).strip()
            r = str(row[1]).strip()
            t = str(row[2]).strip()
            if h and r and t:
                triples.append((h, r, t))
    return triples


def build_kg(triples):
    entity_to_idx = {}
    relation_to_idx = {}

    def get_entity_id(name):
        if name not in entity_to_idx:
            entity_to_idx[name] = len(entity_to_idx)
        return entity_to_idx[name]

    def get_relation_id(name):
        if name not in relation_to_idx:
            relation_to_idx[name] = len(relation_to_idx)
        return relation_to_idx[name]

    edges = []
    for h, r, t in triples:
        hid = get_entity_id(h)
        tid = get_entity_id(t)
        rid = get_relation_id(r)
        edges.append((hid, rid, tid))

    num_entities = len(entity_to_idx)
    adjacency = np.zeros((num_entities, num_entities), dtype=np.uint8)
    adjacency_rel = defaultdict(list)

    for hid, rid, tid in edges:
        adjacency[hid, tid] = 1
        adjacency_rel[(hid, tid)].append(rid)

    idx_to_entity = {idx: name for name, idx in entity_to_idx.items()}

    kg_data = {
        "adjacency": adjacency,
        "entity_to_idx": entity_to_idx,
        "idx_to_entity": idx_to_entity,
        "relation_to_idx": relation_to_idx,
        "adjacency_rel": dict(adjacency_rel),
    }
    return kg_data


def build_token_entity_vocab(entity_to_idx, tokenizer_name):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    vocab = {}
    for entity, entity_id in entity_to_idx.items():
        pieces = tokenizer.encode(entity, add_special_tokens=False)
        if len(pieces) == 0:
            continue
        first_piece = int(pieces[0])
        if first_piece not in vocab:
            vocab[first_piece] = int(entity_id)
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Build KG pickle and entity vocab from triples")
    parser.add_argument("--triples", type=str, required=True, help="Path to triples file (head\trel\ttail or CSV)")
    parser.add_argument("--out_kg", type=str, default="data/kg/mmfn_kg.pkl", help="Output KG pickle path")
    parser.add_argument("--out_vocab", type=str, default="data/kg/entity_vocab.pkl", help="Output token_id->entity_id vocab path")
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="Tokenizer for token-id vocab")
    args = parser.parse_args()

    triples = read_triples(args.triples)
    if len(triples) == 0:
        raise RuntimeError("No valid triples found. Expected at least 3 columns per line.")

    kg_data = build_kg(triples)

    os.makedirs(os.path.dirname(args.out_kg), exist_ok=True)
    with open(args.out_kg, "wb") as f:
        pickle.dump(kg_data, f)

    vocab = build_token_entity_vocab(kg_data["entity_to_idx"], args.tokenizer)
    with open(args.out_vocab, "wb") as f:
        pickle.dump(vocab, f)

    print(f"KG saved: {args.out_kg}")
    print(f"Entity vocab saved: {args.out_vocab}")
    print(f"#entities={len(kg_data['entity_to_idx'])}, #relations={len(kg_data['relation_to_idx'])}, #vocab={len(vocab)}")


if __name__ == "__main__":
    main()
