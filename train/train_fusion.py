# -*- coding: utf-8 -*-
r"""
V2 阶段8(优化): 融合层 + 消融实验 —— 校准后概率 + 学习权重/bias
================================================================
在分支概率校准（Platt）之后进行 logit 空间融合：

    fused_logit = w * logit(p_syntax_cal) + (1-w) * logit(p_lexical_cal) + bias
    p_fusion    = sigmoid(fused_logit)

搜索目标：
  - 主指标 AUC（排序能力）
  - 同 AUC 时取 Brier Score 更小（概率可信度）

输入 : reports/v2/calibrated_probs.npz（校准后 val/test 概率）
输出 : reports/v2/fusion_report.json
       结构: {fusion_method, best_w_syntax, best_bias, val_best_auc, val_best_brier,
              test: {auc, brier, accuracy, f1, human_fp_rate, ai_recall},
              ablation: {syntax_only, lexical_only, fusion}}
"""
import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score, brier_score_loss

_BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
REPORT = os.path.join(REPORTS, "fusion_report.json")
PROBS = os.path.join(REPORTS, "calibrated_probs.npz")


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


def metrics(y_true, prob):
    pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "human_fp_rate": float(fp / (fp + tn)),
        "ai_recall": float(tp / (tp + fn)),
        "n": int(len(y_true)),
    }


def main():
    z = np.load(PROBS)
    y_v, y_t = z["val_labels"], z["test_labels"]
    logit_s_v = to_logit(z["syntax_val"])
    logit_l_v = to_logit(z["lexical_val"])
    logit_s_t = to_logit(z["syntax_test"])
    logit_l_t = to_logit(z["lexical_test"])

    # ---- 网格搜索 (w_syntax, bias)，目标: AUC 优先，Brier 次之 ----
    best = None
    for w in np.arange(0.0, 1.001, 0.05):
        for bias in np.arange(-0.5, 0.501, 0.1):
            fused = sigmoid(w * logit_s_v + (1 - w) * logit_l_v + bias)
            auc = roc_auc_score(y_v, fused)
            brier = brier_score_loss(y_v, fused)
            key = (auc, -brier)  # AUC 大优先；AUC 相同时 Brier 小优先
            if best is None or key > best[0]:
                best = (key, w, bias, auc, brier)
    (_, _), best_w, best_bias, best_auc, best_brier = best
    print(f"验证集最优: w_syntax={best_w:.2f} bias={best_bias:+.2f} "
          f"AUC={best_auc:.4f} Brier={best_brier:.4f}", flush=True)

    # ---- 测试集评估 ----
    fused_t = sigmoid(best_w * logit_s_t + (1 - best_w) * logit_l_t + best_bias)
    m_syntax = metrics(y_t, z["syntax_test"])
    m_lex = metrics(y_t, z["lexical_test"])
    m_fusion = metrics(y_t, fused_t)

    report = {
        "fusion_method": "校准后 logit 空间线性加权 + bias，网格搜索 (w, bias)",
        "best_w_syntax": round(best_w, 2),
        "best_bias": round(best_bias, 2),
        "val_best_auc": round(best_auc, 4),
        "val_best_brier": round(best_brier, 4),
        "ablation": {
            "syntax_only": m_syntax,
            "lexical_only": m_lex,
            "fusion": m_fusion,
        },
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("========== 消融对比 (test, 校准后) ==========", flush=True)
    for name, m in report["ablation"].items():
        print(f"{name:14s} AUC={m['auc']:.4f} Brier={m['brier']:.4f} acc={m['accuracy']:.4f} "
              f"F1={m['f1']:.4f} 人类误报={m['human_fp_rate']:.4f} AI检出={m['ai_recall']:.4f}", flush=True)
    print(f"报告: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
