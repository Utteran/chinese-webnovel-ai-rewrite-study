# -*- coding: utf-8 -*-
"""将有效的500条总结后重写配对展开为 Transformer 分类数据，并按书划分。"""
import os
import json
import random
from collections import Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_BASE, "..", "data", "valid_pairs_summary_500.jsonl")
OUT = os.path.join(_BASE, "..", "data", "transformer_data.jsonl")
SEED = 2026
random.seed(SEED)

pairs = [json.loads(line) for line in open(SRC, encoding="utf-8")]
books = list({p["book"] for p in pairs})
random.shuffle(books)
n = len(books)
n_train = max(1, int(n * 0.7))
n_val = max(1, int(n * 0.15))
split = {b: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, b in enumerate(books)}

rows = []
for p in pairs:
    s = split[p["book"]]
    rows.append({"id": p["id"] + "_human", "pair_id": p["id"], "book": p["book"], "text": p["original"], "label": 0, "split": s})
    rows.append({"id": p["id"] + "_ai", "pair_id": p["id"], "book": p["book"], "text": p["rewritten"], "label": 1, "split": s})

with open(OUT, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"配对数: {len(pairs)} | 文本数: {len(rows)} | 书数: {n}")
print("书籍划分:", Counter(split.values()))
print("文本划分:", Counter(r["split"] for r in rows))
print("标签:", Counter(r["label"] for r in rows))
