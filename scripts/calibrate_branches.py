# -*- coding: utf-8 -*-
r"""
V2 阶段9(优化): 分支概率校准 —— Platt Scaling
================================================
目标：把句式/词汇两个分支的输出从"模型分数"校准为"可用概率"，
使数值可解释（如 0.7 ≈ 70% 证据强度），并为最终分数聚合打基础。

方法：Platt Scaling（sigmoid 校准）
    z      = logit(p)              （原始概率转 logit）
    p_cal  = sigmoid(A * z + B)    （验证集上拟合 A、B 两个参数）
验证集仅 122 样本，2 参数拟合安全；评估用独立 test 集。

校准与评估指标：
  - Brier Score（越小越好）
  - ECE（Expected Calibration Error，分箱校准误差）

输入 : reports/v2/syntax_probs.npz, reports/v2/lexical_probs.npz（原始概率）
输出 :
  - reports/v2/calibration.json
        每分支: {A, B, val/test 校准前后 Brier/ECE}
  - reports/v2/calibrated_probs.npz
        校准后 val/test 概率（供融合层与聚合验证使用）
"""
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
REPORT = os.path.join(REPORTS, "calibration.json")
OUT_NPZ = os.path.join(REPORTS, "calibrated_probs.npz")


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def fit_platt(p, y):
    """在 (logit(p), y) 上拟合逻辑回归 → (A, B)，即 Platt 校准参数。"""
    lr = LogisticRegression(solver="liblinear", C=1e4)
    lr.fit(to_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(p, A, B):
    return sigmoid(A * to_logit(p) + B)


def ece(y_true, prob, bins=10):
    """Expected Calibration Error：按概率分箱，|平均置信度 - 实际频率| 的加权均值。"""
    edges = np.linspace(0, 1, bins + 1)
    total, n = 0.0, len(y_true)
    for i in range(bins):
        if i == bins - 1:
            m = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            m = (prob >= edges[i]) & (prob < edges[i + 1])
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y_true[m].mean()
        total += (m.sum() / n) * abs(conf - acc)
    return float(total)


def load_probs(path):
    z = np.load(path)
    return (z["val_probs"], z["val_labels"], z["test_probs"], z["test_labels"])


def main():
    s_v, s_y_v, s_t, s_y_t = load_probs(os.path.join(REPORTS, "syntax_probs.npz"))
    l_v, l_y_v, l_t, l_y_t = load_probs(os.path.join(REPORTS, "lexical_probs.npz"))
    print(f"数据: val={len(s_y_v)} test={len(s_y_t)}", flush=True)

    calibration = {}
    cals = {}
    for name, p_v, y_v, p_t, y_t in (
        ("syntax", s_v, s_y_v, s_t, s_y_t),
        ("lexical", l_v, l_y_v, l_t, l_y_t),
    ):
        A, B = fit_platt(p_v, y_v)
        p_v_cal = apply_platt(p_v, A, B)
        p_t_cal = apply_platt(p_t, A, B)

        # 校准前后指标（test 集）
        before = {"brier": float(brier_score_loss(y_t, p_t)), "ece": ece(y_t, p_t)}
        after = {"brier": float(brier_score_loss(y_t, p_t_cal)), "ece": ece(y_t, p_t_cal)}

        calibration[name] = {
            "A": A, "B": B,
            "val": {"brier_before": float(brier_score_loss(y_v, p_v)),
                    "brier_after": float(brier_score_loss(y_v, p_v_cal)),
                    "ece_before": ece(y_v, p_v), "ece_after": ece(y_v, p_v_cal)},
            "test": {"brier_before": before["brier"], "brier_after": after["brier"],
                     "ece_before": before["ece"], "ece_after": after["ece"]},
        }
        cals[name] = (p_v_cal, p_t_cal)
        print(f"[{name:8s}] A={A:.3f} B={B:.3f} | test Brier {before['brier']:.4f}->{after['brier']:.4f} "
              f"| test ECE {before['ece']:.4f}->{after['ece']:.4f}", flush=True)

    json.dump(calibration, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 保存校准后概率（融合层/聚合验证复用）
    z_s = np.load(os.path.join(REPORTS, "syntax_probs.npz"))
    z_l = np.load(os.path.join(REPORTS, "lexical_probs.npz"))
    np.savez(OUT_NPZ,
             val_ids=z_s["val_ids"], test_ids=z_s["test_ids"],
             val_labels=s_y_v, test_labels=s_y_t,
             syntax_val=cals["syntax"][0], syntax_test=cals["syntax"][1],
             lexical_val=cals["lexical"][0], lexical_test=cals["lexical"][1])
    print(f"报告: {REPORT}")
    print(f"校准后概率: {OUT_NPZ}")


if __name__ == "__main__":
    main()
