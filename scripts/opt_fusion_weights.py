# -*- coding: utf-8 -*-
r"""
V2 阶段10(实验): 两分支评分占比(w_syntax)优化实验
==================================================
目标：系统性实验「句式分支 vs 词汇分支」在 logit 空间融合中的权重占比 w，
     验证默认等权(w=0.5)是否为最优，并尝试按段落字数分桶的自适应权重。

对比方案：
  A. baseline     : w=0.5, bias=0（当前线上配置）
  B. 全局细粒度   : w∈[0,1] 步长0.01 × bias∈[-0.4,0.4] 步长0.05，val 上网格搜索
  C. 长度分桶自适应: 按段落字数分桶，每桶在 val 上学最优 w（bias 固定 0）
                    test 按桶套用对应权重

稳健性：test 上对 AUC 做 bootstrap 95% CI，判断差异是否显著。

输入 : reports/v2/calibrated_probs.npz
       data/v2/valid_pairs_style_500.jsonl（id → 段落长度）
输出 : reports/v2/weight_opt_report.json
"""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, brier_score_loss

_BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
DATA = os.path.join(_BASE, "..", "data", "v2")
PROBS = os.path.join(REPORTS, "calibrated_probs.npz")
PAIRS = os.path.join(DATA, "valid_pairs_style_500.jsonl")
REPORT = os.path.join(REPORTS, "weight_opt_report.json")

RNG = np.random.default_rng(42)


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def fused_logit(s_cal, l_cal, w, bias=0.0):
    return w * to_logit(s_cal) + (1 - w) * to_logit(l_cal) + bias


def metrics(y_true, prob):
    pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "human_fp_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "ai_recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "n": int(len(y_true)),
    }


def bootstrap_auc(y_true, prob, n_boot=2000):
    """test 段级 AUC 的 bootstrap 95% CI（有放回抽样）。"""
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(set(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], prob[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return {"boot_auc_lo": float(lo), "boot_auc_hi": float(hi)}


def grid_search(y_v, ls_v, ll_v):
    """val 上网格搜索 (w, bias)：AUC 优先，同 AUC 取 Brier 小（与 train_fusion 一致）。"""
    best = None
    for w in np.arange(0.0, 1.0001, 0.01):
        logit = fused_logit(ls_v, ll_v, w, 0.0)
        base_auc = roc_auc_score(y_v, sigmoid(logit))
        for bias in np.arange(-0.4, 0.4001, 0.05):
            fused = sigmoid(logit + bias)
            auc = roc_auc_score(y_v, fused)
            brier = brier_score_loss(y_v, fused)
            key = (auc, -brier)
            if best is None or key > best[0]:
                best = (key, w, bias, auc, brier)
    (_, _), best_w, best_bias, best_auc, best_brier = best
    return float(best_w), float(best_bias), float(best_auc), float(best_brier)


def main():
    z = np.load(PROBS)
    y_v, y_t = z["val_labels"], z["test_labels"]
    ls_v, ll_v = z["syntax_val"], z["lexical_val"]
    ls_t, ll_t = z["syntax_test"], z["lexical_test"]

    # id → 段落长度
    meta = {}
    for r in (json.loads(l) for l in open(PAIRS, encoding="utf-8")):
        for src, key in (("original", "_original"), ("rewritten", "_rewritten")):
            meta[r["id"] + key] = len(r[src])
    len_v = np.array([meta.get(i, 0) for i in z["val_ids"]])
    len_t = np.array([meta.get(i, 0) for i in z["test_ids"]])

    report = {"data": {"n_val": int(len(y_v)), "n_test": int(len(y_t)),
                       "len_val": [int(len_v.min()), int(np.median(len_v)), int(len_v.max())],
                       "len_test": [int(len_t.min()), int(np.median(len_t)), int(len_t.max())]}}

    # ---- A. baseline: w=0.5, bias=0 ----
    p_v = sigmoid(fused_logit(ls_v, ll_v, 0.5))
    p_t = sigmoid(fused_logit(ls_t, ll_t, 0.5))
    report["baseline"] = {
        "w_syntax": 0.5, "bias": 0.0,
        "val": metrics(y_v, p_v),
        "test": metrics(y_t, p_t),
        "test_auc_ci": bootstrap_auc(y_t, p_t),
    }
    print(f"[A baseline] w=0.5      val_AUC={report['baseline']['val']['auc']:.4f} "
          f"test_AUC={report['baseline']['test']['auc']:.4f} "
          f"CI=({report['baseline']['test_auc_ci']['boot_auc_lo']:.4f}, "
          f"{report['baseline']['test_auc_ci']['boot_auc_hi']:.4f})", flush=True)

    # ---- B. 全局细粒度网格 ----
    bw, bb, bv_auc, bv_brier = grid_search(y_v, ls_v, ll_v)
    p_t_b = sigmoid(fused_logit(ls_t, ll_t, bw, bb))
    report["global_fine"] = {
        "w_syntax": bw, "bias": bb,
        "val": {"auc": bv_auc, "brier": bv_brier},
        "test": metrics(y_t, p_t_b),
        "test_auc_ci": bootstrap_auc(y_t, p_t_b),
    }
    print(f"[B global ] w={bw:.2f} bias={bb:+.2f} val_AUC={bv_auc:.4f} "
          f"test_AUC={report['global_fine']['test']['auc']:.4f} "
          f"CI=({report['global_fine']['test_auc_ci']['boot_auc_lo']:.4f}, "
          f"{report['global_fine']['test_auc_ci']['boot_auc_hi']:.4f})", flush=True)

    # ---- w 响应面（bias=0，每 0.02）----
    resp = []
    for w in np.arange(0.0, 1.0001, 0.02):
        resp.append({
            "w": round(float(w), 2),
            "val_auc": float(roc_auc_score(y_v, sigmoid(fused_logit(ls_v, ll_v, w)))),
            "test_auc": float(roc_auc_score(y_t, sigmoid(fused_logit(ls_t, ll_t, w)))),
        })
    report["w_response"] = resp

    # ---- C. 长度分桶自适应权重（每桶独立搜 w，bias=0）----
    buckets = [("<170", 0, 170), ("170-195", 170, 195), (">=195", 195, 10 ** 9)]
    bucket_info, test_fused = [], []
    for name, lo, hi in buckets:
        iv = np.where((len_v >= lo) & (len_v < hi))[0]
        it = np.where((len_t >= lo) & (len_t < hi))[0]
        if len(iv) >= 15 and len(set(y_v[iv])) == 2:
            wb, _, auc_b, brier_b = grid_search(y_v[iv], ls_v[iv], ll_v[iv])
            p_t_bk = sigmoid(fused_logit(ls_t[it], ll_t[it], wb))
            test_fused.extend(p_t_bk.tolist())
            test_idx = it.tolist()
        else:
            # 桶内 val 样本不足：退化为全局细粒度权重
            wb = bw
            p_t_bk = sigmoid(fused_logit(ls_t[it], ll_t[it], wb))
            test_fused.extend(p_t_bk.tolist())
            test_idx = it.tolist()
            auc_b, brier_b = bv_auc, bv_brier
        bucket_info.append({
            "bucket": name, "len_range": [lo, hi],
            "n_val": int(len(iv)), "n_test": int(len(it)),
            "best_w_syntax": round(wb, 2),
            "val_auc": round(auc_b, 4), "val_brier": round(brier_b, 4),
            "fallback": None if len(iv) >= 15 and len(set(y_v[iv])) == 2 else "global_fine",
        })
        print(f"[C bucket {name:8s}] n_val={len(iv):3d} n_test={len(it):3d} w={wb:.2f} "
              f"val_AUC={auc_b:.4f}", flush=True)

    # 按桶拼接回原始顺序
    p_t_c = np.empty_like(p_t)
    for info in bucket_info:
        pass
    order = []
    for _, lo, hi in buckets:
        it = np.where((len_t >= lo) & (len_t < hi))[0]
        order.extend(it.tolist())
    p_t_c[order] = test_fused
    report["length_adaptive"] = {
        "buckets": bucket_info,
        "test": metrics(y_t, p_t_c),
        "test_auc_ci": bootstrap_auc(y_t, p_t_c),
    }
    print(f"[C length ] test_AUC={report['length_adaptive']['test']['auc']:.4f} "
          f"CI=({report['length_adaptive']['test_auc_ci']['boot_auc_lo']:.4f}, "
          f"{report['length_adaptive']['test_auc_ci']['boot_auc_hi']:.4f})", flush=True)

    # ---- 结论 ----
    t_base = report["baseline"]["test"]
    t_glob = report["global_fine"]["test"]
    t_len = report["length_adaptive"]["test"]
    report["conclusion"] = {
        "global_better_than_baseline": (t_glob["auc"] - t_base["auc"]) > 0.001,
        "length_better_than_baseline": (t_len["auc"] - t_base["auc"]) > 0.001,
        "global_delta_auc": round(t_glob["auc"] - t_base["auc"], 4),
        "length_delta_auc": round(t_len["auc"] - t_base["auc"], 4),
        "recommended": None,
    }
    best_auc = max(t_base["auc"], t_glob["auc"], t_len["auc"])
    if best_auc == t_len["auc"]:
        report["conclusion"]["recommended"] = "length_adaptive"
    elif best_auc == t_glob["auc"]:
        report["conclusion"]["recommended"] = "global_fine"
    else:
        report["conclusion"]["recommended"] = "baseline"

    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告: {REPORT}", flush=True)
    print(f"推荐方案: {report['conclusion']['recommended']} "
          f"(global ΔAUC={report['conclusion']['global_delta_auc']:+.4f}, "
          f"length ΔAUC={report['conclusion']['length_delta_auc']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
