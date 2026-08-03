# -*- coding: utf-8 -*-
r"""
V2 阶段3: 句法解析 —— 生成句式分支训练序列
============================================
2.0 双分支方案中，句式分支只接收"语法结构信息"，不接收具体词汇。
本脚本用 LTP（本地模型）对有效配对做 分词 + 词性 + 依存解析，
并把每段文本序列化为两种语法表示：

  - pos_seq : 纯词性序列       "r v nh v nh m q v n wp"
  - dep_seq : 词性:依存关系     "r:SBV v:HED nh:DBL v:VOB ..."

让 Transformer 从序列中隐式学习"句式节奏 / 从句组织 / 成分搭配"等结构特征，
从而捕捉 AI 尽力模仿真人文风后仍残留的句法痕迹。

输入 : data/v2/valid_pairs_style_500.jsonl
输出 : data/v2/syntax_pairs_500.jsonl
       每行: {id, book, label_src(original/rewritten),
              sentence_count, token_count,
              pos_seq, dep_seq}

依赖 : LTP 模型位于 model/ltp_small（本地，含 tokenizer.json）
       兼容补丁: transformers>=5 移除了 batch_encode_plus，
       LTP 内部仍调用它，加载后通过 __call__ 包装绕过。
"""
import os
import re
import sys
import json
import types
from collections import Counter

# ---------------- 配置 ----------------
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2")
SRC = os.path.join(DATA, "valid_pairs_style_500.jsonl")
OUT = os.path.join(DATA, "syntax_pairs_500.jsonl")
LTP_DIR = os.path.join(_BASE, "..", "model", "ltp_small")

SENT_SPLIT = re.compile(r"(?<=[。！？…；])")


def load_ltp(model_dir=LTP_DIR):
    """加载 LTP 并打上 transformers>=5 兼容补丁。"""
    from ltp import LTP
    ltp = LTP(model_dir)
    tok = ltp.tokenizer
    if not hasattr(tok, "batch_encode_plus"):
        # transformers 5.x 移除 batch_encode_plus，__call__ 提供等价功能
        def batch_encode_plus(self, *args, **kwargs):
            return self(*args, **kwargs)
        tok.batch_encode_plus = types.MethodType(batch_encode_plus, tok)
    return ltp


def split_sents(text):
    """按句末标点切分句子。"""
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]


def parse_text(ltp, text):
    """解析一段文本 → (pos_seq, dep_seq, sent_count, token_count)
    整段句子一次性批量送入 LTP（pipeline 内部按 batch 推理），
    避免逐句调用导致的多次前向开销。"""
    sents = split_sents(text)
    out = ltp.pipeline(sents, tasks=["cws", "pos", "dep"])
    pos_parts, dep_parts = [], []
    for i in range(len(sents)):
        pos = out.pos[i]
        labels = out.dep[i]["label"]
        pos_parts.append(" ".join(pos))
        dep_parts.append(" ".join(f"{p}:{rel}" for p, rel in zip(pos, labels)))
    pos_seq = " || ".join(pos_parts)
    dep_seq = " || ".join(dep_parts)
    return pos_seq, dep_seq, len(sents), sum(len(p.split()) for p in pos_parts)


def main():
    ltp = load_ltp()
    recs = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    print(f"解析 {len(recs)} 条有效配对（原文+重写，共 {len(recs) * 2} 段）", flush=True)

    rows = []
    for i, r in enumerate(recs, 1):
        for label_src, text in (("original", r["original"]), ("rewritten", r["rewritten"])):
            pos_seq, dep_seq, n_sent, n_tok = parse_text(ltp, text)
            rows.append({
                "id": r["id"],
                "book": r["book"],
                "label_src": label_src,
                "sentence_count": n_sent,
                "token_count": n_tok,
                "pos_seq": pos_seq,
                "dep_seq": dep_seq,
            })
        if i % 50 == 0:
            print(f"进度 {i}/{len(recs)}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 统计
    pos_tokens = set()
    dep_tokens = set()
    for row in rows:
        pos_tokens.update(row["pos_seq"].replace(" || ", " ").split())
        dep_tokens.update(row["dep_seq"].replace(" || ", " ").split())
    print(f"完成: {len(rows)} 段")
    print(f"词性标签数: {len(pos_tokens)}  词性:依存组合数: {len(dep_tokens)}")
    print(f"样本示例 dep_seq:\n{rows[0]['dep_seq'][:200]}")


if __name__ == "__main__":
    main()
