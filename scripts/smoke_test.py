# -*- coding: utf-8 -*-
"""网页 API 冒烟测试：原文(人类)应低分，AI 重写应高分，混合文本验证存在性聚合。"""
import json
import requests

pairs = [json.loads(l) for l in open("data/v2/valid_pairs_style_500.jsonl", encoding="utf-8")]
p = pairs[0]

for name, t in [("人类(原文)", p["original"]), ("AI(重写)", p["rewritten"])]:
    r = requests.post("http://127.0.0.1:8000/api/detect", json={"text": t}, timeout=120).json()
    print("---", name, "---")
    print("整体:", r["overall_score"], "| 存在概率:", r["ai_presence"],
          "| 痕迹强度:", r["ai_strength"], "| 判定:", r["verdict"]["label"],
          "| 不确定:", r["uncertain"], r["notes"])
    for s in r["segments"]:
        print("  段", round(s["score"], 3), "(句式", round(s["syntax_score"], 3),
              "/ 词汇", round(s["lexical_score"], 3), "| 证据", s["evidence"], ")")

# 混合文本：人类段 + AI 段（各一段），验证存在性聚合不再依赖固定 AI_SEG_BOOST
p2 = pairs[1]
mix = p["original"] + p2["rewritten"]
r = requests.post("http://127.0.0.1:8000/api/detect", json={"text": mix}, timeout=120).json()
print("--- 混合(人类+AI) ---")
print("整体:", r["overall_score"], "| 存在概率:", r["ai_presence"],
      "| 痕迹强度:", r["ai_strength"], "| 判定:", r["verdict"]["label"])
for s in r["segments"]:
    print("  段", round(s["score"], 3), "| 字数", s["chars"], "| 证据", s["evidence"], "|", "AI" if s["ai"] else "类人")

# 短文本：验证不确定提示
short = "他推开门，走了进去。"
r = requests.post("http://127.0.0.1:8000/api/detect", json={"text": short}, timeout=120).json()
print("--- 短文本 ---")
print("整体:", r["overall_score"], "| 不确定:", r["uncertain"], "| 提示:", r["notes"])
