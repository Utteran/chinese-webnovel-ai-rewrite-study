# -*- coding: utf-8 -*-
r"""
V2 阶段7(改进): 词汇分支主模型训练 —— 消融最优特征集
=====================================================
依据 reports/v2/ablation_report.json 的消融结论，采用最优特征组合：
  lex + tf + bg（157 维，raw 频率）
即：词汇丰富度 + 高频词 TF + 高频词对 TF。
舍弃了 len（长度捷径）、pos（词性占比，实验证明是负贡献）、fw（功能词表，无独立价值）。

- 数据  : data/v2/lexical_features_v2.csv（train-only 词表，含 split 列）
- 模型  : XGBoost 二分类（参数与消融/旧版完全一致）
- 输出  :
    - model/lexical_xgb_v2/model.json
    - reports/v2/lexical_report.json
    - reports/v2/lexical_probs.npz      （val/test 概率，供融合层）
    - reports/v2/lexical_feature_names.json（线上 app.py 复用：feature_names/top_words/top_bigrams）
"""
import os
import json
import csv
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2")
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
OUT_DIR = os.path.join(_BASE, "..", "model", "lexical_xgb_v2")
CSV_PATH = os.path.join(DATA, "lexical_features_v2.csv")
SETS_PATH = os.path.join(REPORTS, "lexical_feature_sets.json")
REPORT = os.path.join(REPORTS, "lexical_report.json")
PROBS = os.path.join(REPORTS, "lexical_probs.npz")
FEAT_NAMES_OUT = os.path.join(REPORTS, "lexical_feature_names.json")
SEED = 2026

# 消融实验最优特征组合（见 ablation_report.json: lexical_raw）
FEATURE_GROUPS = ["lex", "tf", "bg"]

# XGBoost 超参（小数据谨慎防止过拟合）
PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 800,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": 8,
    "early_stopping_rounds": 50,
}


def load_data():
    """读 v2 CSV + 特征分组清单，按 FEATURE_GROUPS 选列。"""
    sets_cfg = json.load(open(SETS_PATH, encoding="utf-8"))
    groups = sets_cfg["feature_sets"]
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    names = list(rows[0].keys())[4:]  # 跳过 id/book/label/split
    cols = [c for g in FEATURE_GROUPS for c in groups.get(g, [])]
    X = np.array([[float(r[k]) for k in cols] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    split_arr = np.array([r["split"] for r in rows])
    return X, y, split_arr, cols, sets_cfg, rows


def metrics(y_true, prob):
    pred = (prob >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "human_fp_rate": float(fp / (fp + tn)),
        "ai_recall": float(tp / (tp + fn)),
        "n": int(len(y_true)),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    X, y, split_arr, cols, sets_cfg, rows = load_data()
    print(f"样本: {len(y)} | 特征集 {FEATURE_GROUPS} 共 {len(cols)} 维", flush=True)

    mask = {"train": split_arr == "train", "val": split_arr == "val", "test": split_arr == "test"}
    Xtr, ytr = X[mask["train"]], y[mask["train"]]
    Xva, yva = X[mask["val"]], y[mask["val"]]
    Xte, yte = X[mask["test"]], y[mask["test"]]
    print(f"划分: train={len(ytr)} val={len(yva)} test={len(yte)}", flush=True)

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=50)

    val_prob = model.predict_proba(Xva)[:, 1]
    test_prob = model.predict_proba(Xte)[:, 1]
    val_m = metrics(yva, val_prob)
    test_m = metrics(yte, test_prob)
    print(f"最终 test: AUC={test_m['auc']:.4f} acc={test_m['accuracy']:.4f} F1={test_m['f1']:.4f} "
          f"人类误报={test_m['human_fp_rate']:.4f} AI检出={test_m['ai_recall']:.4f}", flush=True)

    model.save_model(os.path.join(OUT_DIR, "model.json"))

    # 保存 val/test 概率（融合层使用）
    np.savez(PROBS,
             val_ids=ids_of(rows, mask["val"]), val_probs=val_prob, val_labels=yva,
             test_ids=ids_of(rows, mask["test"]), test_probs=test_prob, test_labels=yte)

    # 线上词表文件（app.py 复用：feature_names + top_words/top_bigrams）
    json.dump({"feature_names": cols,
               "top_words": sets_cfg["top_words"],
               "top_bigrams": sets_cfg["top_bigrams"],
               "note": "消融最优特征集 lex+tf+bg (157维), train-only 词表"},
              open(FEAT_NAMES_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 特征重要性 Top-20
    imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:20]
    imp = [(k, float(v)) for k, v in imp]

    report = {
        "branch": "lexical_xgboost",
        "feature_set": FEATURE_GROUPS,
        "input": f"lexical_features_v2.csv ({len(cols)} 维, 消融最优组合)",
        "params": PARAMS,
        "train": len(ytr), "val": len(yva), "test": len(yte),
        "val": val_m, "test": test_m,
        "feature_importance_top20": imp,
        "best_iteration": int(model.best_iteration),
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"模型: {OUT_DIR}/ | 报告: {REPORT}", flush=True)


def ids_of(rows, mask):
    """提取 mask 对应的样本 id。"""
    import numpy as np
    return np.array([r["id"] for r, m in zip(rows, mask) if m])


if __name__ == "__main__":
    main()
