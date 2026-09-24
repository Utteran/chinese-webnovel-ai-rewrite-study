# -*- coding: utf-8 -*-
"""Read-only integrity audit for the checked-in V1 and V2 research artifacts."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(rows, key, name):
    result = {}
    for row in rows:
        ident = key(row)
        require(ident not in result, f"{name}: duplicate ID {ident}")
        result[ident] = row
    return result


def check_manifest(path, books):
    document = json.loads(path.read_text(encoding="utf-8"))
    mapping = document["book_split"]
    require(set(mapping) == set(books), f"{path}: book set differs from the pairs")
    require(document["n_books"] == len(mapping), f"{path}: n_books is wrong")
    require(set(mapping.values()) == {"train", "val", "test"},
            f"{path}: expected train, val, and test")
    return mapping


def check_exact_overlap(samples, name):
    locations = defaultdict(set)
    for text, split in samples:
        require(isinstance(text, str) and bool(text.strip()), f"{name}: empty text")
        locations[text].add(split)
    require(not any(len(splits) > 1 for splits in locations.values()),
            f"{name}: identical text occurs in multiple splits")


def audit_v1(root):
    data = root / "data"
    pairs = keyed(read_jsonl(data / "valid_pairs_summary_500.jsonl"),
                  lambda row: row["id"], "V1 pairs")
    mapping = check_manifest(data / "summary_book_splits.json",
                             (row["book"] for row in pairs.values()))
    rows = keyed(read_jsonl(data / "transformer_data.jsonl"),
                 lambda row: row["id"], "V1 training rows")
    expected = {f"{ident}_{suffix}" for ident in pairs for suffix in ("human", "ai")}
    require(set(rows) == expected, "V1 training rows do not match pair IDs")
    samples = []
    for ident, pair in pairs.items():
        for suffix, field, label in (("human", "original", 0), ("ai", "rewritten", 1)):
            row = rows[f"{ident}_{suffix}"]
            require(row["pair_id"] == ident and row["book"] == pair["book"]
                    and row["label"] == label and row["split"] == mapping[pair["book"]]
                    and row["text"] == pair[field], f"V1 mismatch: {ident}_{suffix}")
            samples.append((row["text"], row["split"]))
    check_exact_overlap(samples, "V1")
    return {"pairs": len(pairs), "rows": len(rows),
            "splits": dict(Counter(row["split"] for row in rows.values()))}


def audit_v2(root):
    data = root / "data" / "v2"
    pairs = keyed(read_jsonl(data / "valid_pairs_style_500.jsonl"),
                  lambda row: row["id"], "V2 pairs")
    mapping = check_manifest(data / "splits.json",
                             (row["book"] for row in pairs.values()))
    syntax = keyed(read_jsonl(data / "syntax_pairs_500.jsonl"),
                   lambda row: (row["id"], row["label_src"]), "V2 syntax")
    training = keyed(read_jsonl(data / "syntax_train_data.jsonl"),
                     lambda row: row["id"], "V2 training rows")
    with (data / "lexical_features_v2.csv").open(encoding="utf-8", newline="") as handle:
        lexical = keyed(csv.DictReader(handle), lambda row: row["id"], "V2 lexical rows")
    expected = {f"{ident}_{source}" for ident in pairs
                for source in ("original", "rewritten")}
    require(set(training) == expected and set(lexical) == expected,
            "V2 training and lexical IDs must match the pairs")
    require(set(syntax) == {(ident, source) for ident in pairs
                            for source in ("original", "rewritten")},
            "V2 syntax IDs must match the pairs")
    samples = []
    for ident, pair in pairs.items():
        for source, label in (("original", 0), ("rewritten", 1)):
            sample_id = f"{ident}_{source}"
            parsed = syntax[(ident, source)]
            row = training[sample_id]
            features = lexical[sample_id]
            split = mapping[pair["book"]]
            require(parsed["book"] == pair["book"] and bool(parsed["dep_seq"].strip()),
                    f"V2 syntax mismatch: {sample_id}")
            require(row["pair_id"] == ident and row["book"] == pair["book"]
                    and row["label"] == label and row["split"] == split
                    and row["text"] == parsed["dep_seq"],
                    f"V2 training mismatch: {sample_id}")
            require(features["book"] == pair["book"]
                    and int(features["label"]) == label and features["split"] == split,
                    f"V2 lexical mismatch: {sample_id}")
            samples.append((pair[source], split))
    check_exact_overlap(samples, "V2")
    return {"pairs": len(pairs), "rows": len(training),
            "splits": dict(Counter(row["split"] for row in training.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps({"v1": audit_v1(args.root), "v2": audit_v2(args.root)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
