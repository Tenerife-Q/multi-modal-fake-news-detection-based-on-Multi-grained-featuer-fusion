import argparse
import json
from pathlib import Path

import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def remove_conflict_keys(train_df: pd.DataFrame, test_df: pd.DataFrame, key_col: str, label_col: str):
    if key_col not in train_df.columns or key_col not in test_df.columns or label_col not in train_df.columns:
        return train_df, test_df, {
            "enabled": False,
            "reason": f"missing columns: key={key_col} or label={label_col}",
            "conflict_keys": 0,
            "train_removed": 0,
            "test_removed": 0,
        }

    t = train_df.copy()
    v = test_df.copy()
    t["__clean_key"] = normalize_series(t[key_col])
    v["__clean_key"] = normalize_series(v[key_col])

    all_df = pd.concat(
        [t[["__clean_key", label_col]], v[["__clean_key", label_col]]],
        axis=0,
        ignore_index=True,
    )
    all_df = all_df[all_df["__clean_key"] != ""]
    all_df[label_col] = pd.to_numeric(all_df[label_col], errors="coerce")
    all_df = all_df.dropna(subset=[label_col])

    nunique = all_df.groupby("__clean_key")[label_col].nunique()
    conflict_keys = set(nunique[nunique > 1].index.tolist())

    train_before = len(t)
    test_before = len(v)
    if conflict_keys:
        t = t[~t["__clean_key"].isin(conflict_keys)]
        v = v[~v["__clean_key"].isin(conflict_keys)]

    t = t.drop(columns=["__clean_key"])
    v = v.drop(columns=["__clean_key"])

    return t, v, {
        "enabled": True,
        "conflict_keys": len(conflict_keys),
        "train_removed": train_before - len(t),
        "test_removed": test_before - len(v),
    }


def dedup_within_split(df: pd.DataFrame, key_col: str):
    if key_col not in df.columns:
        return df, {"enabled": False, "reason": f"missing column: {key_col}", "removed": 0}

    d = df.copy()
    d["__clean_key"] = normalize_series(d[key_col])
    before = len(d)
    non_empty = d["__clean_key"] != ""

    keep_mask = pd.Series(True, index=d.index)
    keep_mask.loc[non_empty] = ~d.loc[non_empty, "__clean_key"].duplicated(keep="first")
    d = d[keep_mask].drop(columns=["__clean_key"])

    return d, {"enabled": True, "removed": before - len(d)}


def drop_cross_split_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    key_col: str,
    where: str,
):
    if key_col not in train_df.columns or key_col not in test_df.columns:
        return train_df, test_df, {
            "enabled": False,
            "reason": f"missing column: {key_col}",
            "overlap_keys": 0,
            "train_removed": 0,
            "test_removed": 0,
        }

    t = train_df.copy()
    v = test_df.copy()
    t["__clean_key"] = normalize_series(t[key_col])
    v["__clean_key"] = normalize_series(v[key_col])

    train_keys = set(t.loc[t["__clean_key"] != "", "__clean_key"].tolist())
    test_keys = set(v.loc[v["__clean_key"] != "", "__clean_key"].tolist())
    overlap = train_keys & test_keys

    train_before = len(t)
    test_before = len(v)
    if where == "train":
        t = t[~t["__clean_key"].isin(overlap)]
    elif where == "test":
        v = v[~v["__clean_key"].isin(overlap)]
    elif where == "both":
        t = t[~t["__clean_key"].isin(overlap)]
        v = v[~v["__clean_key"].isin(overlap)]

    t = t.drop(columns=["__clean_key"])
    v = v.drop(columns=["__clean_key"])

    return t, v, {
        "enabled": where != "none",
        "overlap_keys": len(overlap),
        "train_removed": train_before - len(t),
        "test_removed": test_before - len(v),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Clean GossipCop train/test splits.")
    parser.add_argument("--train-in", type=str, default="gossip/train_gossipcop.csv")
    parser.add_argument("--test-in", type=str, default="gossip/test_gossipcop.csv")
    parser.add_argument("--train-out", type=str, default="gossip/train_gossipcop.clean.csv")
    parser.add_argument("--test-out", type=str, default="gossip/test_gossipcop.clean.csv")
    parser.add_argument("--report-out", type=str, default="gossip/clean_report.json")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--url-col", type=str, default="url")
    parser.add_argument("--title-col", type=str, default="title")
    parser.add_argument(
        "--overlap-drop",
        type=str,
        default="train",
        choices=["none", "train", "test", "both"],
        help="When cross-split overlap exists, remove from which split.",
    )
    parser.add_argument(
        "--disable-title-overlap-clean",
        action="store_true",
        help="Disable title-level overlap cleaning.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train_in = Path(args.train_in)
    test_in = Path(args.test_in)
    train_out = Path(args.train_out)
    test_out = Path(args.test_out)
    report_out = Path(args.report_out)

    train_df = pd.read_csv(train_in, low_memory=False)
    test_df = pd.read_csv(test_in, low_memory=False)

    report = {
        "input": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "config": {
            "label_col": args.label_col,
            "url_col": args.url_col,
            "title_col": args.title_col,
            "overlap_drop": args.overlap_drop,
            "clean_title_overlap": not args.disable_title_overlap_clean,
        },
    }

    train_df, test_df, step_conflict = remove_conflict_keys(
        train_df, test_df, key_col=args.url_col, label_col=args.label_col
    )
    report["step_remove_url_label_conflicts"] = step_conflict

    train_df, step_train_dedup = dedup_within_split(train_df, key_col=args.url_col)
    test_df, step_test_dedup = dedup_within_split(test_df, key_col=args.url_col)
    report["step_dedup_url_train"] = step_train_dedup
    report["step_dedup_url_test"] = step_test_dedup

    train_df, test_df, step_url_overlap = drop_cross_split_overlap(
        train_df, test_df, key_col=args.url_col, where=args.overlap_drop
    )
    report["step_drop_url_overlap"] = step_url_overlap

    if not args.disable_title_overlap_clean:
        train_df, test_df, step_title_overlap = drop_cross_split_overlap(
            train_df, test_df, key_col=args.title_col, where=args.overlap_drop
        )
    else:
        step_title_overlap = {"enabled": False, "reason": "disabled by flag"}
    report["step_drop_title_overlap"] = step_title_overlap

    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    report["output"] = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_out": str(train_out),
        "test_out": str(test_out),
    }

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
