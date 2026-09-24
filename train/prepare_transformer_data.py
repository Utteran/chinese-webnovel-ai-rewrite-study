# -*- coding: utf-8 -*-
"""Expand V1 rewrite pairs using the locked, published book split."""

import argparse
import json
from collections import Counter
from pathlib import Path

from prepare_v2_splits import load_or_make_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "valid_pairs_summary_500.jsonl"
OUT = ROOT / "data" / "transformer_data.jsonl"
MANIFEST = ROOT / "data" / "summary_book_splits.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-split", action="store_true",
                        help="replace the published split; invalidates existing model reports")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    with SRC.open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle if line.strip()]
    split, created = load_or_make_split(
        (pair["book"] for pair in pairs), manifest=MANIFEST,
        new_split=args.new_split, seed=args.seed
    )
    rows = []
    for pair in pairs:
        for suffix, field, label in (("human", "original", 0), ("ai", "rewritten", 1)):
            rows.append({
                "id": f"{pair['id']}_{suffix}",
                "pair_id": pair["id"],
                "book": pair["book"],
                "text": pair[field],
                "label": label,
                "split": split[pair["book"]],
            })

    if created:
        MANIFEST.write_text(
            json.dumps({"book_split": split, "n_books": len(split), "seed": args.seed},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"pairs={len(pairs)} books={len(split)} split_manifest={'new' if created else 'reused'}")
    print("book splits:", dict(Counter(split.values())))
    print("sample splits:", dict(Counter(row["split"] for row in rows)))


if __name__ == "__main__":
    main()
