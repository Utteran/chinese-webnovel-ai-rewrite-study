# -*- coding: utf-8 -*-
"""纯文本 Transformer AI 味分类器训练。输入仅为 text + label，不使用人工统计特征。"""
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
DATA = os.path.join(_BASE, "..", "data", "transformer_data.jsonl")
OUT_DIR = os.path.join(_BASE, "..", "model", "transformer_classifier_500")
REPORT = os.path.join(_BASE, "transformer_500_report.json")
SEED = 2026
MAX_LEN = 256
BATCH = 16
EPOCHS = 4
LR = 2e-5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"设备: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})", flush=True)

class TextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        self.enc = tokenizer([r["text"] for r in rows], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

def evaluate(model, loader):
    model.eval(); probs=[]; labels=[]
    with torch.no_grad():
        for batch, y in loader:
            batch={k:v.to(DEVICE) for k,v in batch.items()}
            out=model(**batch)
            probs.extend(F.softmax(out.logits, dim=1)[:,1].cpu().tolist())
            labels.extend(y.tolist())
    probs=np.array(probs); labels=np.array(labels); pred=(probs>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(labels,pred).ravel()
    return {"auc":float(roc_auc_score(labels,probs)),"accuracy":float(accuracy_score(labels,pred)),"f1":float(f1_score(labels,pred)),"human_fp_rate":float(fp/(fp+tn)),"ai_recall":float(tp/(tp+fn)),"n":len(labels),"probs":probs,"labels":labels}

def main():
    rows=[json.loads(l) for l in open(DATA,encoding="utf-8")]
    train=[r for r in rows if r["split"]=="train"]; val=[r for r in rows if r["split"]=="val"]; test=[r for r in rows if r["split"]=="test"]
    print(f"数据: train={len(train)} val={len(val)} test={len(test)}",flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    model=AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=2).to(DEVICE)
    train_dl=DataLoader(TextDataset(train,tokenizer),batch_size=BATCH,shuffle=True)
    val_dl=DataLoader(TextDataset(val,tokenizer),batch_size=BATCH)
    test_dl=DataLoader(TextDataset(test,tokenizer),batch_size=BATCH)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.01)
    total_steps=EPOCHS*len(train_dl); step=0; best_auc=-1; best_state=None; history=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); total_loss=0
        for batch,y in train_dl:
            batch={k:v.to(DEVICE) for k,v in batch.items()}; y=y.to(DEVICE)
            opt.zero_grad(); loss=model(**batch,labels=y).loss; loss.backward(); opt.step()
            total_loss+=loss.item(); step+=1
            if step%10==0 or step==total_steps: print(f"step {step}/{total_steps} epoch={epoch} loss={loss.item():.4f}",flush=True)
        metrics=evaluate(model,val_dl); metrics.pop("probs"); metrics.pop("labels")
        metrics["epoch"]=epoch; metrics["train_loss"]=total_loss/len(train_dl); history.append(metrics)
        print(f"epoch {epoch}/{EPOCHS} loss={metrics['train_loss']:.4f} val_auc={metrics['auc']:.4f} val_acc={metrics['accuracy']:.4f} human_fp={metrics['human_fp_rate']:.4f}",flush=True)
        if metrics["auc"]>best_auc: best_auc=metrics["auc"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state); test_m=evaluate(model,test_dl); val_m=evaluate(model,val_dl)
    os.makedirs(OUT_DIR,exist_ok=True); model.save_pretrained(OUT_DIR); tokenizer.save_pretrained(OUT_DIR)
    report={"model":MODEL_NAME,"data":DATA,"device":str(DEVICE),"pairs":len(rows)//2,"train":len(train),"val":len(val),"test":len(test),"best_val_auc":best_auc,"val":{"auc":val_m["auc"],"accuracy":val_m["accuracy"],"f1":val_m["f1"],"human_fp_rate":val_m["human_fp_rate"],"ai_recall":val_m["ai_recall"]},"test":{"auc":test_m["auc"],"accuracy":test_m["accuracy"],"f1":test_m["f1"],"human_fp_rate":test_m["human_fp_rate"],"ai_recall":test_m["ai_recall"]},"history":history}
    json.dump(report,open(REPORT,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(f"最终 test: AUC={test_m['auc']:.4f} acc={test_m['accuracy']:.4f} F1={test_m['f1']:.4f} 人类误报={test_m['human_fp_rate']:.4f} AI检出={test_m['ai_recall']:.4f}",flush=True)
    print(f"模型: {OUT_DIR}/ | 报告: {REPORT}",flush=True)
if __name__=="__main__": main()
