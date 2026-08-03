# -*- coding: utf-8 -*-
"""优化后端到端耗时测试：单段(164字) + 长文本(约1600字，10段)。"""
import json
import time
import requests

pairs = [json.loads(l) for l in open("data/v2/valid_pairs_style_500.jsonl", encoding="utf-8")]

# 单段
t0 = time.time()
r1 = requests.post("http://127.0.0.1:8000/api/detect", json={"text": pairs[0]["original"]}, timeout=120)
t1 = time.time()
print(f"单段 (164字): {t1-t0:.3f}s -> 分数 {r1.json()['overall_score']}")

# 长文本：拼接 10 段原文
long_text = "".join(p["original"] for p in pairs[:10])
t0 = time.time()
r2 = requests.post("http://127.0.0.1:8000/api/detect", json={"text": long_text}, timeout=120)
t1 = time.time()
print(f"长文本 ({len(long_text)}字): {t1-t0:.3f}s -> 分数 {r2.json()['overall_score']}, 分段数 {len(r2.json()['segments'])}")
