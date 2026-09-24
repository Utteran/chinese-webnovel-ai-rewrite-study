# -*- coding: utf-8 -*-
r"""
V2 阶段6: 句式分支 Transformer 训练
====================================
输入为语法序列（dep_seq：词性:依存关系），不含具体词汇，
让模型隐式学习 AI 重写后残留的"句式节奏 / 从句组织 / 成分搭配"痕迹。

- 数据  : data/v2/syntax_train_data.jsonl（已按书划分）
- 模型  : hfl/chinese-roberta-wwm-ext 微调（与 1.0 同架构便于对比）
- 输出  : model/syntax_transformer_v2/ + reports/v2/syntax_report.json
          并保存 val/test 预测概率（供融合层使用）
"""
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2", "syntax_train_data.jsonl")
OUT_DIR = os.path.join(_BASE, "..", "model", "syntax_transformer_v2")
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
REPORT = os.path.join(REPORTS, "syntax_report.json")
SEED = 2026
MAX_LEN = 256
BATCH = 16
EPOCHS = 6
LR = 1e-5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"设备: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})", flush=True)


class SyntaxDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        self.enc = tokenizer([r["text"] for r in rows], max_length=MAX_LEN, truncation=True,
                             padding="max_length", return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}, self.labels[i]


def evaluate(model, loader):
    model.eval(); probs = []; labels = []
    with torch.no_grad():
        for batch, y in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            probs.extend(F.softmax(out.logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(y.tolist())
    probs = np.array(probs); labels = np.array(labels); pred = (probs >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred)),
        "human_fp_rate": float(fp / (fp + tn)),
        "ai_recall": float(tp / (tp + fn)),
        "n": len(labels),
        "probs": probs,
        "labels": labels,
        "ids": loader.dataset._ids,
    }


def main():
    os.makedirs(REPORTS, exist_ok=True)
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"数据: train={len(train)} val={len(val)} test={len(test)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    # 记录 id 供概率输出使用
    def make_ds(data):
        ds = SyntaxDataset(data, tokenizer)
        ds._ids = [r["id"] for r in data]
        return ds

    train_dl = DataLoader(make_ds(train), batch_size=BATCH, shuffle=True)
    val_dl = DataLoader(make_ds(val), batch_size=BATCH)
    test_dl = DataLoader(make_ds(test), batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.01)
    total_steps = EPOCHS * len(train_dl); step = 0; best_auc = -1; best_state = None; history = []
    for epoch in range(1, EPOCHS + 1):
        model.train(); total_loss = 0
        for batch, y in train_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}; y = y.to(DEVICE)
            opt.zero_grad(); loss = model(**batch, labels=y).loss; loss.backward(); opt.step()
            total_loss += loss.item(); step += 1
            if step % 20 == 0 or step == total_steps:
                print(f"step {step}/{total_steps} epoch={epoch} loss={loss.item():.4f}", flush=True)
        m = evaluate(model, val_dl)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_dl),
                        "val_auc": m["auc"], "val_acc": m["accuracy"],
                        "val_human_fp": m["human_fp_rate"], "val_ai_recall": m["ai_recall"]})
        print(f"epoch {epoch}/{EPOCHS} loss={history[-1]['train_loss']:.4f} "
              f"val_auc={m['auc']:.4f} val_acc={m['accuracy']:.4f} "
              f"human_fp={m['human_fp_rate']:.4f} ai_recall={m['ai_recall']:.4f}", flush=True)
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_m = evaluate(model, val_dl)
    test_m = evaluate(model, test_dl)

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR); tokenizer.save_pretrained(OUT_DIR)

    # 保存 val/test 概率（融合层使用）
    np.savez(os.path.join(REPORTS, "syntax_probs.npz"),
             val_ids=val_m["ids"], val_probs=val_m["probs"], val_labels=val_m["labels"],
             test_ids=test_m["ids"], test_probs=test_m["probs"], test_labels=test_m["labels"])

    report = {
        "branch": "syntax_transformer",
        "input": "dep_seq (pos:deprel)",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "train": len(train), "val": len(val), "test": len(test),
        "best_val_auc": best_auc,
        "val": {k: v for k, v in val_m.items() if k not in ("probs", "labels", "ids")},
        "test": {k: v for k, v in test_m.items() if k not in ("probs", "labels", "ids")},
        "history": history,
    }
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"最终 test: AUC={test_m['auc']:.4f} acc={test_m['accuracy']:.4f} F1={test_m['f1']:.4f} "
          f"人类误报={test_m['human_fp_rate']:.4f} AI检出={test_m['ai_recall']:.4f}", flush=True)
    print(f"模型: {OUT_DIR}/ | 报告: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
