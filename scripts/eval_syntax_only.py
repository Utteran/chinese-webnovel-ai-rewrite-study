# -*- coding: utf-8 -*-
r"""
句式分支单独评估 + 可视化
==========================
仅用句式 Transformer（dep_seq 语法序列）在测试集上评估，
输出指标报告与可视化图表到 reports/v2/syntax_eval/：

  - roc.png        ROC 曲线
  - pr.png         PR 曲线
  - dist.png       人类/AI 概率分布直方图
  - confusion.png  混淆矩阵
  - threshold.png  阈值-指标曲线
  - errors.txt     Top 误报（人类判AI）与漏检（AI判人类）样例

输入: reports/v2/syntax_probs.npz（已保存的测试集预测）
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix, accuracy_score,
                             f1_score)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

_BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
OUT_DIR = os.path.join(REPORTS, "syntax_eval")
os.makedirs(OUT_DIR, exist_ok=True)


def load_meta():
    """id -> (原始文本, 书, 类型) 映射，用于错误样例展示。"""
    meta = {}
    pairs = [json.loads(l) for l in open(os.path.join(_BASE, "..", "data", "v2",
                                                      "valid_pairs_style_500.jsonl"), encoding="utf-8")]
    for p in pairs:
        meta[p["id"] + "_original"] = {"text": p["original"], "book": p["book"], "kind": "原文"}
        meta[p["id"] + "_rewritten"] = {"text": p["rewritten"], "book": p["book"], "kind": "AI重写"}
    return meta


def main():
    z = np.load(os.path.join(REPORTS, "syntax_probs.npz"))
    ids, probs, labels = z["test_ids"], z["test_probs"], z["test_labels"]
    meta = load_meta()

    # ---- 指标 ----
    pred = (probs >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    metrics = {
        "auc": float(roc_auc_score(labels, probs)),
        "ap": float(average_precision_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred)),
        "human_fp_rate": float(fp / (fp + tn)),
        "ai_recall": float(tp / (tp + fn)),
        "n": int(len(labels)),
    }
    print("==== 仅句式分支 · 测试集 ====")
    print(f"AUC={metrics['auc']:.4f}  AP={metrics['ap']:.4f}  Acc={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}")
    print(f"人类误报={metrics['human_fp_rate']:.4f} ({fp}/{fp + tn})  "
          f"AI检出={metrics['ai_recall']:.4f} ({tp}/{tp + fn})")

    # ---- ROC ----
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(fpr, tpr, lw=2, color="#e05252",
            label=f"句式 Transformer (AUC={metrics['auc']:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax.set_xlabel("假正率（人类误报）"); ax.set_ylabel("真正率（AI检出）")
    ax.set_title("仅句式特征 · ROC 曲线")
    ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "roc.png"), dpi=150); plt.close(fig)

    # ---- PR ----
    prec, rec, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(rec, prec, lw=2, color="#e05252",
            label=f"句式 Transformer (AP={metrics['ap']:.4f})")
    ax.set_xlabel("召回率（AI检出）"); ax.set_ylabel("精确率")
    ax.set_title("仅句式特征 · PR 曲线")
    ax.legend(loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "pr.png"), dpi=150); plt.close(fig)

    # ---- 分布直方图 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 26)
    ax.hist(probs[labels == 0], bins=bins, alpha=.7, color="#4a8fe7", label="人类原文 (label=0)", edgecolor="white")
    ax.hist(probs[labels == 1], bins=bins, alpha=.7, color="#e05252", label="AI 风格重写 (label=1)", edgecolor="white")
    ax.axvline(.5, color="#222", ls="--", lw=1, label="判定阈值 0.5")
    ax.set_xlabel("句式分支 AI 概率"); ax.set_ylabel("样本数")
    ax.set_title("仅句式特征 · 分数分布")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "dist.png"), dpi=150); plt.close(fig)

    # ---- 混淆矩阵 ----
    fig, ax = plt.subplots(figsize=(5, 4.6))
    cm = confusion_matrix(labels, pred)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["人类(预测)", "AI(预测)"])
    ax.set_yticklabels(["人类(真实)", "AI(真实)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_title("仅句式特征 · 混淆矩阵 (test)")
    fig.colorbar(im, shrink=.85)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "confusion.png"), dpi=150); plt.close(fig)

    # ---- 阈值曲线 ----
    thr = np.linspace(0.05, 0.95, 91)
    accs, fps, recs = [], [], []
    for t in thr:
        p = (probs >= t).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(labels, p).ravel()
        accs.append((tp2 + tn2) / len(labels))
        fps.append(fp2 / (fp2 + tn2))
        recs.append(tp2 / (tp2 + fn2))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thr, accs, label="准确率", color="#4a8fe7")
    ax.plot(thr, recs, label="AI检出率", color="#e05252")
    ax.plot(thr, fps, label="人类误报率", color="#999")
    ax.axvline(.5, color="#222", ls="--", lw=1)
    ax.set_xlabel("判定阈值"); ax.set_ylabel("比率")
    ax.set_title("仅句式特征 · 阈值-指标曲线")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "threshold.png"), dpi=150); plt.close(fig)

    # ---- 错误样例 ----
    lines = ["==== 仅句式分支 测试集错误样例 ====", ""]
    lines.append(f"整体指标: AUC={metrics['auc']:.4f} Acc={metrics['accuracy']:.4f} "
                 f"F1={metrics['f1']:.4f} 人类误报={metrics['human_fp_rate']:.3f} AI检出={metrics['ai_recall']:.3f}")
    lines.append("")
    # 误报：人类原文被判成 AI（label=0, prob 高）
    fp_idx = np.argsort(probs[labels == 0])[::-1][:8]
    lines.append("---- 误报 Top8（人类原文 → 被判 AI）----")
    for i in fp_idx:
        idx = np.where(labels == 0)[0][i]
        m = meta.get(ids[idx], {})
        lines.append(f"[{ids[idx]}] {m.get('kind', '?')} p={probs[idx]:.3f} | {m.get('book', '?')}")
        lines.append(f"  文本: {(m.get('text') or '')[:90]}")
    lines.append("")
    # 漏检：AI 重写被判成人类（label=1, prob 低）
    fn_idx = np.argsort(probs[labels == 1])[:8]
    lines.append("---- 漏检 Top8（AI 重写 → 被判人类）----")
    for i in fn_idx:
        idx = np.where(labels == 1)[0][i]
        m = meta.get(ids[idx], {})
        lines.append(f"[{ids[idx]}] {m.get('kind', '?')} p={probs[idx]:.3f} | {m.get('book', '?')}")
        lines.append(f"  文本: {(m.get('text') or '')[:90]}")
    with open(os.path.join(OUT_DIR, "errors.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n图表已输出到: {OUT_DIR}/")


if __name__ == "__main__":
    main()
