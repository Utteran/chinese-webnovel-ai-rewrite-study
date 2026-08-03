# -*- coding: utf-8 -*-
r"""
V2 阶段5: 数据准备 —— 按书划分 + 句式分支数据展开
==================================================
严格按"书"划分 train/val/test（70/15/15），防止同一作者的写作风格
同时出现在训练与测试中造成泄漏（1.0 已确立的核心原则）。

输出 :
  - data/v2/splits.json              书 → split 映射
  - data/v2/syntax_train_data.jsonl  句式分支分类数据
      每行: {id, pair_id, book, text=dep_seq, label(0原文/1重写), split}
"""
import os
import json
import random
from collections import Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2")
PAIRS = os.path.join(DATA, "valid_pairs_style_500.jsonl")
SYNTAX = os.path.join(DATA, "syntax_pairs_500.jsonl")
OUT_SPLITS = os.path.join(DATA, "splits.json")
OUT_SYNTAX = os.path.join(DATA, "syntax_train_data.jsonl")
SEED = 2026

random.seed(SEED)

pairs = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
syntax = {r["id"] + "_" + r["label_src"]: r for r in (json.loads(l) for l in open(SYNTAX, encoding="utf-8"))}

# ---- 按书划分 ----
books = list({p["book"] for p in pairs})
random.shuffle(books)
n = len(books)
n_train = max(1, int(n * 0.7))
n_val = max(1, int(n * 0.15))
split = {b: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
         for i, b in enumerate(books)}

# ---- 展开句式分类数据 ----
rows = []
for p in pairs:
    s = split[p["book"]]
    for label_src, label in (("original", 0), ("rewritten", 1)):
        key = p["id"] + "_" + label_src
        r = syntax[key]
        rows.append({
            "id": key,
            "pair_id": p["id"],
            "book": p["book"],
            "text": r["dep_seq"],
            "label": label,
            "split": s,
        })

with open(OUT_SPLITS, "w", encoding="utf-8") as f:
    json.dump({"book_split": split, "n_books": n}, f, ensure_ascii=False, indent=2)
with open(OUT_SYNTAX, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"配对数: {len(pairs)} | 书数: {n}")
print("书划分:", dict(Counter(split.values())))
print("样本划分:", dict(Counter(r["split"] for r in rows)))
print("标签分布:", dict(Counter(r["label"] for r in rows)))
