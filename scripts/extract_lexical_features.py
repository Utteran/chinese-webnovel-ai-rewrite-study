# -*- coding: utf-8 -*-
r"""
V2 阶段4(改进): 词汇特征提取 v2 —— 分组特征 + train-only 词表 + 频率变换
========================================================================
相对 v1（原 extract_lexical_features.py）的改进：

  1. 特征按语义分组（len/lex/pos/fw/tf/ltf/bg/lbg），供消融实验自由组合
  2. 高频词表 / 高频词对表 只从 train 集构建（修复 val/test 参与特征空间定义的泄漏）
  3. 词频同时输出两套频率：
       - tf:词 / bg:词对 = 归一化频率 count/N（raw）
       - ltf:词 / lbg:词对 = log1p(count)（压缩高频词主导，改善长尾）
     由消融实验对比哪种频率变换更有效，模型仍是同一个 XGBoost。

特征分组说明：
  - len : 文本长度/句子结构/标点密度（长度捷径嫌疑组，用于消融验证）
  - lex : 词汇丰富度与词长分布（TTR、单/双字词比、词种数等）
  - pos : 词性占比（n/v/a/d/r/p/c/u）
  - fw  : 功能词（虚词/连接词）归一化频次
  - tf  : 高频词 raw 频率     | ltf : 高频词 log 频率
  - bg  : 高频词对 raw 频率   | lbg : 高频词对 log 频率

输入 : data/v2/valid_pairs_style_500.jsonl + data/v2/splits.json
输出 :
  - data/v2/lexical_features_v2.csv
       每行: id, book, label, split, <len组>, <lex组>, <pos组>, <fw组>, <tf/ltf组>, <bg/lbg组>
  - reports/v2/lexical_feature_sets.json
       分组名 -> 特征名列表（供消融训练脚本按组选列）
"""
import os
import re
import json
import math
from collections import Counter

import jieba
import jieba.posseg as pseg

# ---------------- 配置 ----------------
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2")
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
SRC = os.path.join(DATA, "valid_pairs_style_500.jsonl")
SPLITS = os.path.join(DATA, "splits.json")
OUT_CSV = os.path.join(DATA, "lexical_features_v2.csv")
OUT_SETS = os.path.join(REPORTS, "lexical_feature_sets.json")

TOP_WORD = 100        # 全局高频词表规模
TOP_BIGRAM = 50       # 全局高频相邻词对表规模
SENT_SPLIT = re.compile(r"[。！？…；]")

# 功能词表（虚词 / 连接词 / 高频代词）
FUNC_WORDS = [
    "的", "了", "是", "在", "有", "就", "都", "也", "还", "不", "着", "过",
    "把", "被", "向", "从", "于", "与", "和", "或者", "但", "却", "而", "然后",
    "于是", "因为", "所以", "如果", "虽然", "即使", "可是", "不过", "但是",
    "一", "这", "那", "说", "道", "他", "她", "我", "你", "们",
]


def read_records():
    """展开配对为样本，并标注每个样本所属 split（按书划分）。"""
    splits = json.load(open(SPLITS, encoding="utf-8"))["book_split"]
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    rows = []
    for r in recs:
        s = splits[r["book"]]
        rows.append({"id": r["id"] + "_original", "book": r["book"], "split": s,
                     "label": 0, "text": r["original"]})
        rows.append({"id": r["id"] + "_rewritten", "book": r["book"], "split": s,
                     "label": 1, "text": r["rewritten"]})
    return rows


def tokenize(text):
    """一次 pseg 调用同时得到 (词, 词性) 列表，避免 jieba.cut 重复分词。
    注意：pseg.cut 的分词粒度与 jieba.cut 不同（如"两人"→"两/人"），
    因此训练 CSV 与线上推理必须统一使用本函数，保证特征完全一致。"""
    return [(w, t) for w, t in pseg.cut(text)]


def build_train_vocab(rows):
    """只统计 train 集样本的高频词与高频相邻词对，固定词表供全体使用。"""
    word_cnt = Counter()
    bigram_cnt = Counter()
    for row in rows:
        if row["split"] != "train":
            continue
        words = [w for w, _ in tokenize(row["text"])]
        word_cnt.update(words)
        bigram_cnt.update(zip(words, words[1:]))
    top_words = [w for w, _ in word_cnt.most_common(TOP_WORD)]
    top_bigrams = [f"{a}|{b}" for (a, b), _ in bigram_cnt.most_common(TOP_BIGRAM)]
    return top_words, top_bigrams


def raw_tf(count, N):
    return round(count / max(N, 1), 5)


def log_tf(count, N):
    return round(math.log1p(count), 5)


def extract_features(text, top_words, top_bigrams):
    """返回 (特征dict, 组dict: 组名 -> 该组特征名列表)。"""
    feat, group = {}, {}
    L = len(text)
    sents = [s for s in SENT_SPLIT.split(text) if s.strip()]
    sent_lens = [len(s) for s in sents] if sents else [L]
    # 一次 pseg 调用同时得到词与词性（避免 jieba.cut 重复分词，见 tokenize 注释）
    wt = tokenize(text)
    words = [w for w, _ in wt]
    N = len(words)
    word_freq = Counter(words)

    def add(grp, name, value):
        feat[name] = value
        group.setdefault(grp, []).append(name)

    # ---- len 组: 长度 / 句子结构 / 标点密度 ----
    add("len", "char_len", L)
    add("len", "word_count", N)
    add("len", "sent_count", len(sents))
    add("len", "mean_sent_len", round(sum(sent_lens) / len(sent_lens), 3))
    add("len", "std_sent_len",
        round(math.sqrt(sum((x - sum(sent_lens) / len(sent_lens)) ** 2 for x in sent_lens) / len(sent_lens)), 3)
        if len(sent_lens) > 1 else 0.0)
    for ch, name in [("，", "comma"), ("。", "period"), ("！", "excl"), ("？", "ques"),
                     ("…", "ellipsis"), ("；", "semicolon")]:
        add("len", f"{name}_density", round(text.count(ch) / max(L, 1) * 100, 4))
    add("len", "hanzi_ratio", round(len(re.findall(r"[\u4e00-\u9fff]", text)) / max(L, 1), 4))

    # ---- lex 组: 词汇丰富度 / 词长分布 ----
    uniq = set(words)
    add("lex", "uniq_word_count", len(uniq))
    add("lex", "ttr", round(len(uniq) / max(N, 1), 4))
    if N:
        wlens = [len(w) for w in words]
        add("lex", "single_word_ratio", round(wlens.count(1) / N, 4))
        add("lex", "double_word_ratio", round(wlens.count(2) / N, 4))
        add("lex", "multi_word_ratio", round(sum(1 for x in wlens if x >= 3) / N, 4))
        add("lex", "mean_word_len", round(sum(wlens) / N, 4))
        add("lex", "std_word_len",
            round(math.sqrt(sum((x - sum(wlens) / N) ** 2 for x in wlens) / N), 4) if N > 1 else 0.0)
    else:
        for k in ("single_word_ratio", "double_word_ratio", "multi_word_ratio",
                  "mean_word_len", "std_word_len"):
            add("lex", k, 0.0)

    # ---- pos 组: 词性占比（复用上面 pseg 结果，不再二次调用）----
    pos_cnt = Counter(t for _, t in wt)
    pos_total = sum(pos_cnt.values())
    for cls in ["n", "v", "a", "d", "r", "p", "c", "u"]:
        c = sum(v for k, v in pos_cnt.items() if k.startswith(cls))
        add("pos", f"pos_{cls}", round(c / max(pos_total, 1), 4))

    # ---- fw 组: 功能词归一化频次 ----
    for w in FUNC_WORDS:
        add("fw", f"fw:{w}", round(word_freq.get(w, 0) / max(N, 1), 5))

    # ---- tf / ltf 组: 高频词 raw 与 log 频率 ----
    for w in top_words:
        c = word_freq.get(w, 0)
        add("tf", f"tf:{w}", raw_tf(c, N))
        add("ltf", f"ltf:{w}", log_tf(c, N))

    # ---- bg / lbg 组: 高频词对 raw 与 log 频率 ----
    bigram_freq = Counter(zip(words, words[1:]))
    for bg in top_bigrams:
        a, b = bg.split("|")
        c = bigram_freq.get((a, b), 0)
        add("bg", f"bg:{bg}", raw_tf(c, N))
        add("lbg", f"lbg:{bg}", log_tf(c, N))

    return feat, group


def main():
    rows = read_records()
    print(f"样本数: {len(rows)}（{len(rows) // 2} 对配对 × 原文/重写）", flush=True)
    print(f"划分: train={sum(1 for r in rows if r['split']=='train')} "
          f"val={sum(1 for r in rows if r['split']=='val')} "
          f"test={sum(1 for r in rows if r['split']=='test')}", flush=True)

    # 词表只由 train 集构建（修复泄漏）
    top_words, top_bigrams = build_train_vocab(rows)
    print(f"train-only 高频词表 {len(top_words)} 个, 高频词对 {len(top_bigrams)} 个", flush=True)

    all_feats, all_groups = [], {}
    for r in rows:
        feat, group = extract_features(r["text"], top_words, top_bigrams)
        all_feats.append(feat)
        for grp, names in group.items():
            all_groups.setdefault(grp, []).extend(names)
    # 去重并保持首次出现顺序
    all_groups = {g: list(dict.fromkeys(ns)) for g, ns in all_groups.items()}

    ordered_groups = ["len", "lex", "pos", "fw", "tf", "ltf", "bg", "lbg"]
    feat_names = [n for g in ordered_groups for n in all_groups.get(g, [])]
    print(f"特征维度: {len(feat_names)} | 分组: "
          + ", ".join(f"{g}={len(all_groups[g])}" for g in ordered_groups), flush=True)

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("id,book,label,split," + ",".join(feat_names) + "\n")
        for r, feats in zip(rows, all_feats):
            vals = [r["id"], f'"{r["book"]}"', str(r["label"]), r["split"]]
            vals += [str(feats[k]) for k in feat_names]
            f.write(",".join(vals) + "\n")

    with open(OUT_SETS, "w", encoding="utf-8") as f:
        json.dump({"feature_sets": all_groups, "feature_names": feat_names,
                   "top_words": top_words, "top_bigrams": top_bigrams,
                   "note": "词表仅由 train 集构建"}, f, ensure_ascii=False, indent=2)

    print(f"完成: {OUT_CSV}")
    print(f"分组清单: {OUT_SETS}")


if __name__ == "__main__":
    main()
