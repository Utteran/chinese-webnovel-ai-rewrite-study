# -*- coding: utf-8 -*-
r"""
阶段2+3: AI 重写配对数据生成器
- 从 train 集随机抽 500 段
- 所有样本统一使用 A 类: 总结 → 根据总结重写
- 调用 DeepSeek API, 带重试 + 断点续跑
输出: pairs_summary_500.jsonl —— 每行 {id, original, summary, rewritten, model, temperature, prompt_version, status}
"""
import os
import re
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ---- 配置 ----
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY，例如: $env:DEEPSEEK_API_KEY='sk-xxx'")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

_BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_BASE, "..", "data", "dataset_enriched.jsonl")
OUT = os.path.join(_BASE, "..", "data", "pairs_summary_500.jsonl")
SAMPLE_N = 500                    # 总结后重写数据点数量
TYPE = "summary_rewrite"
SEED = 2026
MAX_WORKERS = 8
RETRY = 3
TARGET_LEN = 200

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---- Prompt 库 ----
PROMPT_V = "rewrite_v1"

SUMMARY_PROMPT = (
    "请用不超过50字概括下面这段网文正文的故事情节大意。"
    "只输出大意本身，不要任何开场白、解释或标点之外的修饰。\n\n原文：\n{text}"
)

REWRITE_PROMPT_A = (
    "根据下面这段故事大意，重新写一段约200字的网文正文。要求：\n"
    "1. 完全基于大意展开，不得添加原文没有的情节、人物或设定\n"
    "2. 语言自然流畅，像真正的网文作者写的，避免排比堆砌和空洞形容词\n"
    "3. 只输出正文，不要任何解释或前缀\n\n大意：\n{summary}"
)

REWRITE_PROMPT_B = (
    "请将下面这段网文正文改写为含义相同的新段落，约200字。\n"
    "要求：不得遗漏重要情节，不得添加新的人物或设定，语言像真人网文作者。\n"
    "只输出改写后的正文，不要任何解释。\n\n原文：\n{text}"
)

REWRITE_PROMPT_C = (
    "下面是网文正文。请保留情节和人物关系，改写成更符合网络小说风格的段落，约200字。\n"
    "要求：可适度调整表达，不得改变情节走向，不得添加新设定。\n"
    "只输出改写后的正文，不要任何解释。\n\n原文：\n{text}"
)

CHAT_WORDS_RE = re.compile(r"（?(?:以下是|这是|好的|明白了|没问题|收到|好的，|你好，)?(?:改写|重写|生成|根据)?[^。]{0,6}(?:后的)?(?:文本|正文|内容|结果|如下)[:：]?）?", re.I)


def call_llm(prompt, temperature=0.9, max_tokens=500):
    for attempt in range(RETRY):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == RETRY - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def clean_output(s):
    """去掉模型常见套话/前缀"""
    s = CHAT_WORDS_RE.sub("", s).strip()
    # 去掉首尾多余的引号
    s = re.sub(r"^[「『\"'“]+|[」』\"'”]+$", "", s.strip())
    return s.strip()


def gen_type_a(text):
    """总结 → 重写"""
    summary = call_llm(SUMMARY_PROMPT.format(text=text), temperature=0.3, max_tokens=120)
    if not summary:
        return None, None
    summary = clean_output(summary)
    rewritten = call_llm(REWRITE_PROMPT_A.format(summary=summary), temperature=0.9, max_tokens=500)
    return summary, clean_output(rewritten) if rewritten else None


def gen_type_b(text):
    """直接改写"""
    rewritten = call_llm(REWRITE_PROMPT_B.format(text=text), temperature=0.9, max_tokens=500)
    return None, clean_output(rewritten) if rewritten else None


def gen_type_c(text):
    """指定文体"""
    rewritten = call_llm(REWRITE_PROMPT_C.format(text=text), temperature=0.9, max_tokens=500)
    return None, clean_output(rewritten) if rewritten else None


def gen_one(item):
    """统一执行: 原文 → 总结 → 基于总结重写"""
    text = item["text"]
    try:
        summary, rewritten = gen_type_a(text)
        status = "ok" if rewritten else "empty"
        return {
            "id": item["id"],
            "book": item["book"],
            "type": TYPE,
            "original": text,
            "summary": summary,
            "rewritten": rewritten,
            "model": MODEL,
            "temperature": 0.9,
            "prompt_version": PROMPT_V,
            "status": status,
        }
    except Exception as e:
        return {
            "id": item["id"], "book": item["book"], "type": TYPE,
            "original": text, "summary": None, "rewritten": None,
            "model": MODEL, "temperature": 0.9,
            "prompt_version": PROMPT_V, "status": f"error: {e}",
        }


def main():
    random.seed(SEED)
    items = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    train_items = [d for d in items if d["split"] == "train"]
    sampled = random.sample(train_items, min(SAMPLE_N, len(train_items)))
    print(f"计划: 段落 {len(sampled)}, 类型={TYPE}, 合计 {len(sampled)}")

    # 断点续跑: 读取已完成 id（只认当前输出文件的成功记录）
    done_ids = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                rec = json.loads(l)
                if rec.get("type") == TYPE and rec.get("status") == "ok":
                    done_ids.add(rec["id"])
            except Exception:
                pass

    pending = [d for d in sampled if d["id"] not in done_ids]
    print(f"待生成: {len(pending)}（已完成 {len(done_ids)}）")

    ok = 0
    err = 0
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(gen_one, d): d for d in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if rec["status"] == "ok":
                ok += 1
            else:
                err += 1
            if i % 50 == 0 or i == len(futs):
                print(f"进度 {i}/{len(futs)}: ok={ok} err={err}")

    print(f"完成: 成功 {ok}, 失败/空 {err}")


if __name__ == "__main__":
    main()
