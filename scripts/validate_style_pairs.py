# -*- coding: utf-8 -*-
r"""
V2 阶段2: 风格模仿配对数据质量过滤
====================================
输入 : data/v2/raw_pairs_style_500.jsonl
输出 :
  - data/v2/valid_pairs_style_500.jsonl   通过过滤的有效配对
  - data/v2/invalid_pairs_style_500.jsonl 被剔除的样本（附剔除原因）
  - reports/v2/validation_report.json     过滤统计报告
  - reports/v2/sample_check_50.jsonl      人工抽查样本（50 对）

过滤规则（与 2.0 计划一致）
----------------------------
1. generation_failed  : status 非 ok 或重写为空
2. len_out_of_range   : 重写长度不在 [0.65*L, 1.5*L]（L = 原文长度）
3. high_copy_ratio    : 复制检测（见 copy_ratio），判定为照搬原文而非重写
4. low_hanzi_ratio    : 汉字占比 < 0.70（夹杂异常字符）
5. chat_prefix        : 首部含模型套话/前缀

复制检测说明
------------
中文 200 字段落内常用字重合度高，"最大程度模仿风格"也会大量复用词汇，
因此不能用单字符覆盖率判断"是否照抄"。采用两个更可靠的指标：
- bigram 覆盖率: 重写文本的相邻字对有多少出现在原文中（> 0.55 判定照搬句式）
- 最长公共子串比例: 重写与原文共享的最长连续片段占原文比例（> 0.25 判定复制段落）
"""
import os
import re
import json
from collections import Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data", "v2")
REPORTS = os.path.join(_BASE, "..", "reports", "v2")
SRC = os.path.join(DATA, "raw_pairs_style_500.jsonl")
VALID = os.path.join(DATA, "valid_pairs_style_500.jsonl")
INVALID = os.path.join(DATA, "invalid_pairs_style_500.jsonl")
REPORT = os.path.join(REPORTS, "validation_report.json")
SAMPLE = os.path.join(REPORTS, "sample_check_50.jsonl")

# 规则阈值
LEN_MIN_RATIO = 0.65
LEN_MAX_RATIO = 1.5
BIGRAM_MAX = 0.55
LCS_MAX = 0.25
HANZI_MIN = 0.70
SAMPLE_N = 50

CHAT_RE = re.compile(
    r"(以下是|这是|好的|明白了|没问题|收到|改写后的|重写后的|生成结果|文本如下|内容如下|希望这个|请查收|当然可以)"
)


def bigram_overlap(original, rewritten):
    """重写文本的相邻字符对（bigram）有多少比例出现在原文中。"""
    if not original or not rewritten:
        return 0.0
    orig_bigrams = set(zip(original, original[1:]))
    if not orig_bigrams:
        return 0.0
    hit = sum(1 for bg in zip(rewritten, rewritten[1:]) if bg in orig_bigrams)
    return hit / max(len(rewritten) - 1, 1)


def lcs_ratio(a, b):
    """最长公共子串长度 / 原文长度，衡量是否复制了连续整段。"""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best / max(n, 1)


def check(r):
    reasons = []
    original = r.get("original") or ""
    rewritten = r.get("rewritten") or ""
    L = len(original)

    if r.get("type") != "style_imitation":
        reasons.append("wrong_type")
    if r.get("status") != "ok" or not rewritten:
        reasons.append("generation_failed")
    if L and rewritten:
        if len(rewritten) < LEN_MIN_RATIO * L or len(rewritten) > LEN_MAX_RATIO * L:
            reasons.append(f"len_out_of_range:{len(rewritten)}/{L}")
        if bigram_overlap(original, rewritten) > BIGRAM_MAX:
            reasons.append(f"high_bigram_overlap:{bigram_overlap(original, rewritten):.2f}")
        if lcs_ratio(original, rewritten) > LCS_MAX:
            reasons.append(f"high_lcs_ratio:{lcs_ratio(original, rewritten):.2f}")
    if rewritten and len(re.findall(r"[\u4e00-\u9fff]", rewritten)) / len(rewritten) < HANZI_MIN:
        reasons.append("low_hanzi_ratio")
    if rewritten and CHAT_RE.search(rewritten[:80]):
        reasons.append("chat_prefix")
    return reasons


os.makedirs(REPORTS, exist_ok=True)
recs = [json.loads(line) for line in open(SRC, encoding="utf-8")]
valid, invalid = [], []
counts = Counter()
for r in recs:
    reasons = check(r)
    if reasons:
        r["validation_reasons"] = reasons
        invalid.append(r)
        counts.update(reasons)
    else:
        valid.append(r)

for path, data in ((VALID, valid), (INVALID, invalid)):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 人工抽查：均匀抽样有效样本
import random
random.seed(2026)
sample = random.sample(valid, min(SAMPLE_N, len(valid)))
with open(SAMPLE, "w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps({
            "id": r["id"], "book": r["book"],
            "original": r["original"], "rewritten": r["rewritten"],
        }, ensure_ascii=False) + "\n")

valid_len = [len(r["rewritten"]) for r in valid]
orig_len = [len(r["original"]) for r in valid]
report = {
    "source": SRC,
    "total": len(recs),
    "valid": len(valid),
    "invalid": len(invalid),
    "valid_rate": round(len(valid) / len(recs), 4) if recs else None,
    "invalid_reasons": dict(counts),
    "book_count": len(set(r["book"] for r in valid)),
    "rewrite_length": {
        "min": min(valid_len) if valid_len else None,
        "max": max(valid_len) if valid_len else None,
        "mean": round(sum(valid_len) / len(valid_len), 1) if valid_len else None,
    },
    "original_length": {
        "min": min(orig_len) if orig_len else None,
        "max": max(orig_len) if orig_len else None,
        "mean": round(sum(orig_len) / len(orig_len), 1) if orig_len else None,
    },
    "copy_ratio_dist": {
        "bigram_overlap_mean": round(sum(bigram_overlap(r["original"], r["rewritten"]) for r in valid) / len(valid), 3) if valid else None,
        "lcs_ratio_mean": round(sum(lcs_ratio(r["original"], r["rewritten"]) for r in valid) / len(valid), 3) if valid else None,
    },
    "sample_check_file": SAMPLE,
}
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
