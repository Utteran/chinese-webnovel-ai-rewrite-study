# -*- coding: utf-8 -*-
r"""
网页端 AI 味检测器 —— FastAPI 后端（2.0 双分支版）
====================================================
检测流程：文本 → 自动分段 → 每段并行计算两个分支分数
  - 句式分支 : LTP 句法解析(dep_seq) → Transformer 隐式学习句式结构
  - 词汇分支 : jieba 词汇特征(157维消融最优: lex+tf+bg) → XGBoost 隐式学习用词痕迹
  - 融合     : logit 空间线性加权（权重由验证集学习）
复用 scripts/ 下的解析与特征提取模块，确保线上特征与训练一致。

运行: AKTool\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
"""
import os
import sys
import re
import json
import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from parse_syntax import load_ltp, parse_text
from extract_lexical_features import extract_features

MODEL_DIR = os.path.join(BASE_DIR, "model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "v2")

SYNTAX_DIR = os.path.join(MODEL_DIR, "syntax_transformer_v2")
XGB_PATH = os.path.join(MODEL_DIR, "lexical_xgb_v2", "model.json")
FEAT_NAMES = os.path.join(REPORTS_DIR, "lexical_feature_names.json")
CALIBRATION = os.path.join(REPORTS_DIR, "calibration.json")
FUSION_REPORT = os.path.join(REPORTS_DIR, "fusion_report.json")

# 判定档位（对校准后的 AI 参与概率）
THRESHOLDS = [
    (0.35, "偏人类", "text"),
    (0.55, "不确定", "low"),
    (0.75, "疑似 AI", "mid"),
    (1.01, "高度疑似 AI", "high"),
]

# 最终得分聚合参数
EVIDENCE_TAU = 0.5      # 段落证据强度阈值：p 超过该值才算 AI 证据（仅用于逐段 evidence 展示）
SHORT_TEXT = 120        # 短文本下限（低于此提示结果仅供参考）
CONFLICT_TH = 0.45      # 两分支校准分差超过此值提示"分支结论不一致"


def warmup():
    """启动预热：LTP/jieba/Transformer 首次推理开销较大，
    提前跑一次小文本，避免用户首次检测遇到冷启动卡顿。"""
    try:
        parse_text(models["ltp"], "他推开门，走了进去。")
        lexical_score("他推开门，走了进去。")
    except Exception as e:
        print(f"[warmup] 预热失败(忽略): {e}", flush=True)


app = FastAPI(title="文味 Detect 2.0", lifespan=None)


@app.on_event("startup")
def _startup():
    warmup()


class DetectRequest(BaseModel):
    text: str


def load_models():
    """加载 2.0 双分支模型 + LTP + 词表 + 融合权重。"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 句式分支
    tokenizer = AutoTokenizer.from_pretrained(SYNTAX_DIR)
    syntax_model = AutoModelForSequenceClassification.from_pretrained(SYNTAX_DIR)
    syntax_model.to(device)
    syntax_model.eval()

    # 词汇分支
    booster = xgb.Booster()
    booster.load_model(XGB_PATH)

    # 词汇特征词表（与训练时一致）
    feat_cfg = json.load(open(FEAT_NAMES, encoding="utf-8"))
    top_words = feat_cfg["top_words"]
    top_bigrams = feat_cfg["top_bigrams"]
    feature_names = feat_cfg["feature_names"]

    # 分支校准参数（Platt: p_cal = sigmoid(A*logit(p)+B)）
    calib = json.load(open(CALIBRATION, encoding="utf-8"))
    cal_syntax = calib["syntax"]
    cal_lexical = calib["lexical"]

    # 融合参数（校准后 logit 空间线性加权 + bias，验证集学习）
    fusion = json.load(open(FUSION_REPORT, encoding="utf-8"))
    w_syntax = fusion["best_w_syntax"]
    fusion_bias = fusion["best_bias"]

    # LTP（含 transformers>=5 兼容补丁）
    ltp = load_ltp()

    return {
        "device": device, "tokenizer": tokenizer, "syntax_model": syntax_model,
        "booster": booster, "top_words": top_words, "top_bigrams": top_bigrams,
        "feature_names": feature_names, "w_syntax": w_syntax, "fusion_bias": fusion_bias,
        "cal_syntax": cal_syntax, "cal_lexical": cal_lexical, "ltp": ltp,
    }


models = load_models()


def segment(text):
    """按句末标点切句，贪心组装 140~200 字分段。"""
    parts = re.split(r"(?<=[。！？…；])", text)
    sents = [p.strip() for p in parts if p.strip()]
    segs, cur = [], ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) >= 140 and len(cur) + len(s) > 200:
            segs.append(cur)
            cur = s
        else:
            cur += s
    if len(cur) >= 80:
        segs.append(cur)
    return segs


def lexical_score(text):
    """词汇分支：词汇特征(157维消融最优集) → XGBoost → P(AI)。"""
    m = models
    feats, _ = extract_features(text, m["top_words"], m["top_bigrams"])
    vec = np.array([[feats[k] for k in m["feature_names"]]], dtype=np.float32)
    dmat = xgb.DMatrix(vec)
    return float(m["booster"].predict(dmat)[0])


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(np.log(p / (1 - p)))


def platt_calibrate(p, cal):
    """Platt 校准：p_cal = sigmoid(A * logit(p) + B)。"""
    return float(1 / (1 + np.exp(-(cal["A"] * to_logit(p) + cal["B"]))))


def fused_score(s_cal, l_cal, w, bias):
    """校准后概率的 logit 空间线性加权融合 + bias。"""
    logit = w * to_logit(s_cal) + (1 - w) * to_logit(l_cal) + bias
    return float(1 / (1 + np.exp(-logit)))


def score_segments(segs):
    """逐段双分支推理 → 校准 → 融合，返回 (融合分, 句式校准分, 词汇校准分)。
    性能优化：
      1. 句式 Transformer 对所有段落做一次性 batch 推理
      2. 每段 LTP 整段句子批量解析、jieba 一次 pseg 调用"""
    m = models
    # 1) 每段: 句法解析 + 词汇特征
    dep_seqs, l_raw = [], []
    for seg in segs:
        _, dep_seq, _, _ = parse_text(m["ltp"], seg)
        dep_seqs.append(dep_seq)
        feats, _ = extract_features(seg, m["top_words"], m["top_bigrams"])
        vec = np.array([[feats[k] for k in m["feature_names"]]], dtype=np.float32)
        l_raw.append(float(m["booster"].predict(xgb.DMatrix(vec))[0]))
    # 2) 句式 Transformer 批量推理
    if dep_seqs:
        enc = m["tokenizer"](dep_seqs, max_length=256, truncation=True, padding="max_length",
                             return_tensors="pt")
        with torch.no_grad():
            out = m["syntax_model"](**{k: v.to(m["device"]) for k, v in enc.items()})
        s_raw = F.softmax(out.logits, dim=1)[:, 1].tolist()
    else:
        s_raw = []
    # 3) 校准 + 融合
    results = []
    for s, l in zip(s_raw, l_raw):
        s_cal = platt_calibrate(s, m["cal_syntax"])
        l_cal = platt_calibrate(l, m["cal_lexical"])
        p = fused_score(s_cal, l_cal, m["w_syntax"], m["fusion_bias"])
        results.append((p, s_cal, l_cal))
    return results


def evidence(p):
    """段落证据强度：p 超过 EVIDENCE_TAU 的部分映射到 [0,1]。"""
    return float(max(0.0, p - EVIDENCE_TAU) / (1 - EVIDENCE_TAU))


def aggregate_scores(scored, segs):
    """多段 → 最终判定分：按段落字数权值加权平均。
    每段权重 = 该段字数 / 总字数，最终分 = Σ w_i * p_i。
    单段: 直接返回该段概率。"""
    if not scored:
        return 0.0, 0.0
    if len(scored) == 1:
        return scored[0][0], scored[0][0]
    lens = np.array([len(s) for s in segs], dtype=float)
    weights = lens / lens.sum()
    wavg = float(np.dot(np.array([sc[0] for sc in scored]), weights))
    return wavg, wavg


def style_stats(text):
    n = len(text)
    sents = [s for s in re.split(r"[。！？…；]", text) if s.strip()]
    return {
        "chars": n,
        "sentence_count": len(sents),
    }


def verdict(score):
    for th, label, level in THRESHOLDS:
        if score < th:
            return {"label": label, "level": level}
    return {"label": "高度疑似 AI", "level": "high"}


@app.post("/api/detect")
def detect(req: DetectRequest):
    text = (req.text or "").strip()
    if not text:
        return JSONResponse({"error": "请输入文本"}, status_code=400)
    if len(text) > 20000:
        return JSONResponse({"error": "文本过长，请控制在 20000 字以内"}, status_code=400)

    segs = segment(text)
    scored = score_segments(segs)

    # 最终判定分：按段落字数权值加权平均（presence / strength 同值）
    presence, strength = aggregate_scores(scored, segs)
    overall = round(presence, 4)

    segments = [
        {
            "text": s,
            "score": round(float(sc[0]), 4),
            "syntax_score": round(float(sc[1]), 4),
            "lexical_score": round(float(sc[2]), 4),
            "evidence": round(evidence(sc[0]), 4),
            "ai": bool(sc[0] >= 0.5),
            "chars": len(s),
        }
        for s, sc in zip(segs, scored)
    ]

    # 不确定状态：文本过短 或 整体处于中间区间时两分支校准分严重冲突
    # （强证据主导的高/低分场景不提示，避免削弱可信判定）
    uncertain = False
    notes = []
    if len(text) < SHORT_TEXT:
        uncertain = True
        notes.append(f"文本仅 {len(text)} 字，样本过短，结果仅供参考")
    if scored and 0.35 <= overall < 0.75 and abs(scored[0][1] - scored[0][2]) > CONFLICT_TH:
        uncertain = True
        notes.append("句式与词汇分支结论不一致，结果存在不确定性")

    return {
        "overall_score": overall,
        "ai_presence": round(presence, 4),
        "ai_strength": round(strength, 4),
        "verdict": verdict(overall),
        "uncertain": uncertain,
        "notes": notes,
        "stats": style_stats(text),
        "segments": segments,
        "model": "文味 Detect 2.0 · 双分支校准融合 + 字数加权判定 (442对风格模仿训练)",
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
