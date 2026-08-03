# -*- coding: utf-8 -*-
r"""
V2 阶段9(优化)验证: 最终得分方案离线评估
==========================================
在 test 集上验证最终得分链路的三件事：

1. 分支校准效果（Brier/ECE 前后对比，摘要自 calibration.json）
2. 段落级长度分组稳定性（融合校准分按长度分桶的 AUC/误报/检出）
3. 聚合方案对比（按"书"构造文档，比较三种聚合的文档级区分能力）：
     - mean         : 全段简单平均
     - max          : 最大段分数
     - topk_presence: 本文采用的 Top-K 存在性聚合 + 强度加权（final = α·presence+(1-α)·strength）

输入 : reports/v2/calibrated_probs.npz（校准后 test 概率 + 标签 + id）
       reports/v2/fusion_report.json（融合权重）
       data/v2/valid_pairs_style_500.jsonl（id→书→文本长度 映射）
输出 : reports/v2/final_score_eval.json
"""
import os
import json
import math
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2")
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
PROBS = os.path.join(REPORTS, "calibrated_probs.npz")
FUSION = os.path.join(REPORTS, "fusion_report.json")
PAIRS = os.path.join(DATA, "valid_pairs_style_500.jsonl")
REPORT = os.path.join(REPORTS, "final_score_eval.json")

TAU = 0.5
ALPHA = 0.7


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def evidence(p):
    return max(0.0, p - TAU) / (1 - TAU)


def fused_p(s_cal, l_cal, w, bias):
    return float(sigmoid(w * to_logit(s_cal) + (1 - w) * to_logit(l_cal) + bias))


def bucket_metrics(y, p):
    if len(set(y)) < 2:
        return None
    pred = (p >= .5).astype(int)
    tn = ((y == 0) & (pred == 0)).sum(); fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum(); tp = ((y == 1) & (pred == 1)).sum()
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float((tp + tn) / len(y)),
        "human_fp": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "ai_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "n": int(len(y)),
    }


def main():
    z = np.load(PROBS)
    fusion = json.load(open(FUSION, encoding="utf-8"))
    w = fusion["best_w_syntax"]
    bias = fusion["best_bias"]

    test_ids = z["test_ids"]
    y_t = z["test_labels"]
    s_cal = z["syntax_test"]
    l_cal = z["lexical_test"]
    p_fused = np.array([fused_p(s, l, w, bias) for s, l in zip(s_cal, l_cal)])

    # id → (书, 文本长度) 映射
    meta = {}
    for r in (json.loads(l) for l in open(PAIRS, encoding="utf-8")):
        for src, key in (("original", "_original"), ("rewritten", "_rewritten")):
            meta[r["id"] + key] = {"book": r["book"], "len": len(r[src])}
    books = [meta.get(i, {}).get("book", "?") for i in test_ids]
    lens = [meta.get(i, {}).get("len", 0) for i in test_ids]

    # ---- 1. 长度分组 ----
    buckets = {"<120": (0, 120), "120-180": (120, 180), "180-250": (180, 250), ">=250": (250, 10**9)}
    length_buckets = {}
    for name, (lo, hi) in buckets.items():
        m = [(i < hi) & (i >= lo) for i in lens]
        idx = [k for k, v in enumerate(m) if v]
        if len(idx) >= 8 and len(set(y_t[idx])) == 2:
            length_buckets[name] = bucket_metrics(y_t[idx], p_fused[idx])
            b = length_buckets[name]
            print(f"[长度 {name:8s}] n={b['n']:3d} AUC={b['auc']:.4f} 误报={b['human_fp']:.3f} "
                  f"检出={b['ai_recall']:.3f}", flush=True)
        else:
            length_buckets[name] = {"n": len(idx), "note": "样本过少或无两类标签"}
            print(f"[长度 {name:8s}] n={len(idx)} 样本不足，跳过", flush=True)

    # ---- 2. 文档级聚合方案对比（按书构造文档）----
    docs = defaultdict(lambda: {"y": 0, "segs": []})
    for i, (bid, y) in enumerate(zip(books, y_t)):
        docs[bid]["segs"].append((p_fused[i], y))
    doc_list = [{"book": b, "segs": d["segs"], "y": 0 if all(s[1] == 0 for s in d["segs"]) else 1}
                for b, d in docs.items() if len(d["segs"]) >= 2]
    # 标注：若文档内既有 0 又有 1（同书原文+重写段落混在同一文档），按多数/是否含 AI 判
    for doc in doc_list:
        doc["y"] = 1 if any(s[1] == 1 for s in doc["segs"]) else 0

    def topk_presence_score(segs):
        """Top-K 存在性聚合：final = α·presence + (1-α)·strength。"""
        ps = [s[0] for s in segs]
        evs = [evidence(p) for p in ps]
        k = min(5, max(1, math.ceil(len(ps) * 0.3)))
        order = sorted(range(len(ps)), key=lambda i: -evs[i])[:k]
        presence = 1.0 - float(np.prod([1 - evs[i] for i in order]))
        w_sum = sum(evs[i] for i in order)
        strength = float(sum(evs[i] * ps[i] for i in order) / w_sum) if w_sum > 1e-9 else float(np.mean(ps))
        return ALPHA * presence + (1 - ALPHA) * strength

    agg_results = {}
    if len(doc_list) >= 6 and len({d["y"] for d in doc_list}) == 2:
        for name, fn in (
            ("mean", lambda segs: float(np.mean([s[0] for s in segs]))),
            ("max", lambda segs: float(max(s[0] for s in segs))),
            ("topk_presence", topk_presence_score),
        ):
            doc_scores = [fn(d["segs"]) for d in doc_list]
            doc_labels = [d["y"] for d in doc_list]
            if len(set(doc_labels)) == 2:
                agg_results[name] = {
                    "auc": float(roc_auc_score(doc_labels, doc_scores)),
                    "brier": float(brier_score_loss(doc_labels, doc_scores)),
                    "n_docs": len(doc_list),
                }
                print(f"[聚合 {name:14s}] 文档级 AUC={agg_results[name]['auc']:.4f} "
                      f"Brier={agg_results[name]['brier']:.4f} (n={len(doc_list)})", flush=True)
    else:
        agg_results = {"note": "文档级样本不足，跳过聚合对比", "n_docs": len(doc_list)}

    # ---- 3. 汇总 ----
    calib = json.load(open(os.path.join(REPORTS, "calibration.json"), encoding="utf-8"))
    summary = {
        "fusion": {"w_syntax": w, "bias": bias},
        "calibration_test": {k: {"brier_before": v["test"]["brier_before"],
                                 "brier_after": v["test"]["brier_after"],
                                 "ece_before": v["test"]["ece_before"],
                                 "ece_after": v["test"]["ece_after"]}
                             for k, v in calib.items()},
        "segment_level": bucket_metrics(y_t, p_fused),
        "length_buckets": length_buckets,
        "aggregation": agg_results,
        "aggregation_params": {"tau": TAU, "alpha": ALPHA},
    }
    json.dump(summary, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    sl = summary["segment_level"]
    print(f"\n段落级(test): AUC={sl['auc']:.4f} Brier={sl['brier']:.4f} "
          f"误报={sl['human_fp']:.3f} 检出={sl['ai_recall']:.3f} (n={sl['n']})")
    print(f"报告: {REPORT}")


if __name__ == "__main__":
    main()
