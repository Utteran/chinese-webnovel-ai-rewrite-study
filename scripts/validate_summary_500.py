# -*- coding: utf-8 -*-
"""验证 500 对总结后重写数据。"""
import os
import json
import re
from collections import Counter

_BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_BASE, "..", "data")
SRC = os.path.join(DATA, "pairs_summary_500.jsonl")
VALID = os.path.join(DATA, "valid_pairs_summary_500.jsonl")
INVALID = os.path.join(DATA, "invalid_pairs_summary_500.jsonl")
REPORT = os.path.join(DATA, "summary_500_validation_report.json")

CHAT_RE = re.compile(r"(以下是|这是|改写后的|重写后的|生成结果|文本如下|内容如下|希望这个|请查收|好的，|当然可以)")

def check(r):
    reasons = []
    original = r.get("original") or ""
    summary = r.get("summary") or ""
    rewritten = r.get("rewritten") or ""
    if r.get("type") != "summary_rewrite":
        reasons.append("wrong_type")
    if r.get("status") != "ok" or not rewritten:
        reasons.append("generation_failed")
    if not summary:
        reasons.append("empty_summary")
    if len(summary) > 100:
        reasons.append("summary_too_long")
    if len(rewritten) < 120:
        reasons.append(f"rewrite_too_short:{len(rewritten)}")
    if len(rewritten) > 400:
        reasons.append(f"rewrite_too_long:{len(rewritten)}")
    if rewritten and len(re.findall(r"[\u4e00-\u9fff]", rewritten)) / len(rewritten) < 0.7:
        reasons.append("low_hanzi_ratio")
    if CHAT_RE.search(rewritten[:80]):
        reasons.append("chat_prefix")
    # 文本完全重复或极高复制：总结后重写应发生表达变化
    if original and rewritten == original:
        reasons.append("identical_to_original")
    return reasons

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

lengths = [len(r["rewritten"]) for r in valid]
report = {
    "source": SRC,
    "total": len(recs),
    "valid": len(valid),
    "invalid": len(invalid),
    "type_counts": dict(Counter(r.get("type") for r in recs)),
    "status_counts": dict(Counter(r.get("status") for r in recs)),
    "invalid_reasons": dict(counts),
    "valid_rewrite_length": {
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "mean": round(sum(lengths) / len(lengths), 1) if lengths else None,
    },
}
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
