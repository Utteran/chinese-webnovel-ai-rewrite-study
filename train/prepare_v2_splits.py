# -*- coding: utf-8 -*-
"""Build V2 classification rows using a locked, book-level split manifest.

The checked-in split is part of the published experiment. Re-running this
script must not silently change it. Use --new-split only for a new experiment.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v2"
PAIRS = DATA / "valid_pairs_style_500.jsonl"
SYNTAX = DATA / "syntax_pairs_500.jsonl"
OUT_SPLITS = DATA / "splits.json"
OUT_SYNTAX = DATA / "syntax_train_data.jsonl"
SEED = 2026
SPLIT_NAMES = {"train", "val", "test"}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_book_split(books, seed=SEED):
    """Sort before shuffling so hash randomization cannot change the split."""
    ordered = sorted(set(books))
    if len(ordered) < 3:
        raise ValueError("At least three books are needed for train/val/test")
    random.Random(seed).shuffle(ordered)
    n_train = max(1, int(len(ordered) * 0.7))
    n_val = max(1, int(len(ordered) * 0.15))
    return {
        book: "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i, book in enumerate(ordered)
    }


def load_or_make_split(books, manifest=OUT_SPLITS, new_split=False, seed=SEED):
    expected = set(books)
    if manifest.exists() and not new_split:
        saved = json.loads(manifest.read_text(encoding="utf-8"))["book_split"]
        if set(saved) != expected or set(saved.values()) != SPLIT_NAMES:
            raise ValueError("Existing split manifest does not match the books or split names")
        return saved, False
    return make_book_split(expected, seed), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-split", action="store_true",
                        help="replace the published split; invalidates existing model reports")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="seed used only with --new-split or when no manifest exists")
    args = parser.parse_args()

    pairs = read_jsonl(PAIRS)
    syntax_rows = read_jsonl(SYNTAX)
    syntax = {(row["id"], row["label_src"]): row for row in syntax_rows}
    if len(syntax) != len(syntax_rows):
        raise ValueError("Duplicate (id, label_src) in syntax pairs")

    split, created = load_or_make_split(
        (pair["book"] for pair in pairs), new_split=args.new_split, seed=args.seed
    )
    rows = []
    for pair in pairs:
        for source, label in (("original", 0), ("rewritten", 1)):
            parsed = syntax[(pair["id"], source)]
            if parsed["book"] != pair["book"]:
                raise ValueError(f"Book mismatch for {pair['id']} {source}")
            rows.append({
                "id": f"{pair['id']}_{source}",
                "pair_id": pair["id"],
                "book": pair["book"],
                "text": parsed["dep_seq"],
                "label": label,
                "split": split[pair["book"]],
            })

    if len(rows) != 2 * len(pairs) or len(syntax) != len(rows):
        raise ValueError("Pair and syntax row counts differ")
    if created:
        OUT_SPLITS.write_text(
            json.dumps({"book_split": split, "n_books": len(split), "seed": args.seed},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    with OUT_SYNTAX.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"pairs={len(pairs)} books={len(split)} split_manifest={'new' if created else 'reused'}")
    print("book splits:", dict(Counter(split.values())))
    print("sample splits:", dict(Counter(row["split"] for row in rows)))


if __name__ == "__main__":
    main()
