# -*- coding: utf-8 -*-
r"""
V2 阶段7(改进): 词汇分支消融实验 —— 同一 XGBoost，不同特征组合
==============================================================
不更换模型，只改变特征输入，验证各特征组的真实贡献：

  E1 current_raw    : len+lex+pos+fw+tf+bg（近似原 220 维基线，词表已改 train-only）
  E2 current_log    : 同上，但词频用 log1p（ltf+lbg）
  E3 no_length_raw  : 去掉 len 组（长度捷径检验）
  E4 no_length_log  : 同上 + log 词频
  E5 lexical_raw    : lex+tf+bg（词汇丰富度+词频，无词性无长度）
  E6 lexical_log    : 同上 + log 词频
  E7 freq_only_raw  : 纯词频 tf+bg（词语选择是否足够）
  E8 freq_only_log  : 同上 + log 词频
  E9 lexical_pos_raw: lex+pos+tf+bg（词汇+词性+词频）
  E10 lexical_pos_log: 同上 + log 词频

所有实验共用：
  - 同一按书划分（data/v2/splits.json）
  - 同一 XGBoost 超参（train_lexical_xgb.py 的参数，含早停）
  - 同一随机种子

输入 : data/v2/lexical_features_v2.csv + reports/v2/lexical_feature_sets.json
输出 : reports/v2/ablation_report.json（含各实验 val/test 指标）
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
CSV_PATH = os.path.join(DATA, "lexical_features_v2.csv")
SETS_PATH = os.path.join(REPORTS, "lexical_feature_sets.json")
REPORT = os.path.join(REPORTS, "ablation_report.json")
SEED = 2026

# XGBoost 超参（与主模型一致）
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

# 实验定义: (名称, 使用的特征组, 词频模式 raw/log)
EXPERIMENTS = [
    ("current_raw",     ["len", "lex", "pos", "fw", "tf", "bg"], "raw"),
    ("current_log",     ["len", "lex", "pos", "fw", "ltf", "lbg"], "log"),
    ("no_length_raw",   ["lex", "pos", "fw", "tf", "bg"], "raw"),
    ("no_length_log",   ["lex", "pos", "fw", "ltf", "lbg"], "log"),
    ("lexical_raw",     ["lex", "tf", "bg"], "raw"),
    ("lexical_log",     ["lex", "ltf", "lbg"], "log"),
    ("freq_only_raw",   ["tf", "bg"], "raw"),
    ("freq_only_log",   ["ltf", "lbg"], "log"),
    ("lexical_pos_raw", ["lex", "pos", "tf", "bg"], "raw"),
    ("lexical_pos_log", ["lex", "pos", "ltf", "lbg"], "log"),
]


def load_data():
    sets_cfg = json.load(open(SETS_PATH, encoding="utf-8"))["feature_sets"]
    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # 特征矩阵按 CSV 列顺序（跳过 id/book/label/split）
    names = list(rows[0].keys())[4:]
    X = np.array([[float(r[k]) for k in names] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    split_arr = np.array([r["split"] for r in rows])
    return X, y, split_arr, names, sets_cfg


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


def run_experiment(name, groups, freq_mode, X, y, split_arr, names, sets_cfg):
    """按特征组选列并训练，返回实验报告。"""
    cols = []
    for g in groups:
        cols.extend(sets_cfg.get(g, []))
    idx = [names.index(c) for c in cols]
    Xs = X[:, idx]
    mask = {"train": split_arr == "train", "val": split_arr == "val", "test": split_arr == "test"}

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(Xs[mask["train"]], y[mask["train"]],
              eval_set=[(Xs[mask["val"]], y[mask["val"]])], verbose=False)
    val_m = metrics(y[mask["val"]], model.predict_proba(Xs[mask["val"]])[:, 1])
    test_m = metrics(y[mask["test"]], model.predict_proba(Xs[mask["test"]])[:, 1])
    return {
        "name": name, "freq_mode": freq_mode,
        "feature_groups": groups, "n_features": len(cols),
        "best_iteration": int(model.best_iteration),
        "val": val_m, "test": test_m,
    }


def main():
    X, y, split_arr, names, sets_cfg = load_data()
    print(f"样本: {len(y)} | CSV 特征维度: {len(names)}", flush=True)

    results = []
    for name, groups, mode in EXPERIMENTS:
        r = run_experiment(name, groups, mode, X, y, split_arr, names, sets_cfg)
        results.append(r)
        print(f"[{r['name']:16s}] feat={r['n_features']:3d} "
              f"val_AUC={r['val']['auc']:.4f} test_AUC={r['test']['auc']:.4f} "
              f"test_acc={r['test']['accuracy']:.4f} test_F1={r['test']['f1']:.4f} "
              f"人类误报={r['test']['human_fp_rate']:.3f} AI检出={r['test']['ai_recall']:.3f}", flush=True)

    json.dump({"params": PARAMS, "experiments": results},
              open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告: {REPORT}")


if __name__ == "__main__":
    main()
