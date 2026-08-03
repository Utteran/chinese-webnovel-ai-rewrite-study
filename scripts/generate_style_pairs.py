# -*- coding: utf-8 -*-
r"""
V2 阶段1: AI 风格模仿重写数据生成器
====================================
目标：生成 "人类原文 — AI 最大程度模仿原文风格重写" 配对数据（500 对）。
用于学习：AI 尽力模仿真人风格后仍残留的可检测特征（2.0 检测器训练语料）。

设计要点
--------
1. 不走"总结→重写"两步，直接让模型模仿原文风格重写（去掉管道中间态痕迹）
2. 高并发（ThreadPoolExecutor, 默认 100 worker）加速 API 调用
3. 断点续跑：已成功写入的 id 自动跳过，中断后重跑即可续传
4. 实时落盘：每个任务完成立即追加写入 JSONL，主线程统一写文件避免冲突
5. 可复现：固定随机种子，抽样逻辑与 1.0 保持一致

输入 : data/dataset_enriched.jsonl  （1.0 清洗后的 1200 段人类网文）
输出 : data/v2/raw_pairs_style_500.jsonl
       每行: {id, book, type, original, rewritten, model, temperature, prompt_version, status}
"""
import os
import re
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ---------------- 配置 ----------------
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY，例如: $env:DEEPSEEK_API_KEY='sk-xxx'")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"

_BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_BASE, "..", "data", "dataset_enriched.jsonl")
OUT = os.path.join(_BASE, "..", "data", "v2", "raw_pairs_style_500.jsonl")

SAMPLE_N = 500          # 目标配对数量
TYPE = "style_imitation"
PROMPT_V = "style_imitation_v1"
SEED = 2026
MAX_WORKERS = 100       # 并发请求数
RETRY = 3               # 网络/5xx 重试次数
TEMPERATURE = 0.7
MAX_TOKENS = 600

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------------- Prompt ----------------
STYLE_IMITATE_PROMPT = (
    "请重写下面这段网文正文。\n\n"
    "要求：\n"
    "1. 最大程度模仿原文的写作风格：叙事视角、语气、句式节奏、用词习惯、段落结构\n"
    "2. 保留原文的人物、事件、动作顺序和核心信息，不得改变情节\n"
    "3. 不得总结、解释或评价原文\n"
    "4. 不得复制原句，也不得仅做同义词替换\n"
    "5. 输出长度与原文大致相当（约200字）\n"
    "6. 只输出重写后的正文，不要任何前缀、解释或说明\n\n"
    "原文：\n{text}"
)

# 模型常见套话前缀（如 "以下是重写结果："），用于清洗输出
CHAT_WORDS_RE = re.compile(
    r"（?(?:以下是|这是|好的|明白了|没问题|收到)?"
    r"(?:改写|重写|生成)?[^。]{0,8}?(?:后的)?(?:文本|正文|内容|结果|如下)[:：]?）?",
    re.I,
)


# ---------------- 工具函数 ----------------
def call_llm(prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    """调用 DeepSeek，失败自动重试（仅重试网络/5xx 类异常）"""
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
            time.sleep(1.5 * (attempt + 1))
    return None


def clean_output(s):
    """去掉模型常见套话前缀与首尾多余引号"""
    s = CHAT_WORDS_RE.sub("", s).strip()
    s = re.sub(r"^[「『\"'“]+|[」』\"'”]+$", "", s.strip())
    return s.strip()


def gen_one(item):
    """单个样本：原文 → 风格模仿重写，返回完整记录"""
    text = item["text"]
    try:
        rewritten = call_llm(STYLE_IMITATE_PROMPT.format(text=text))
        rewritten = clean_output(rewritten) if rewritten else None
        status = "ok" if rewritten else "empty"
        return {
            "id": item["id"],
            "book": item["book"],
            "type": TYPE,
            "original": text,
            "rewritten": rewritten,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "prompt_version": PROMPT_V,
            "status": status,
        }
    except Exception as e:
        return {
            "id": item["id"],
            "book": item["book"],
            "type": TYPE,
            "original": text,
            "rewritten": None,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "prompt_version": PROMPT_V,
            "status": f"error: {e}",
        }


# ---------------- 主流程 ----------------
def main():
    random.seed(SEED)
    items = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    train_items = [d for d in items if d.get("split") == "train"]
    sampled = random.sample(train_items, min(SAMPLE_N, len(train_items)))
    print(f"计划: 采样段落 {len(sampled)}（来源书数 {len(set(d['book'] for d in sampled))}）", flush=True)

    # 断点续跑：读取已有记录，保留成功条目，重写文件剔除失败残留（避免重复行累积）
    done_ids = set()
    if os.path.exists(OUT):
        keep = []
        for l in open(OUT, encoding="utf-8"):
            try:
                rec = json.loads(l)
            except Exception:
                continue
            if rec.get("type") == TYPE and rec.get("status") == "ok":
                keep.append(rec)
                done_ids.add(rec["id"])
        if len(keep) != sum(1 for _ in open(OUT, encoding="utf-8")):
            with open(OUT, "w", encoding="utf-8") as f:
                for rec in keep:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"已清理 {len(keep)} 条成功记录（移除失败残留）", flush=True)

    pending = [d for d in sampled if d["id"] not in done_ids]
    print(f"待生成: {len(pending)}（已完成 {len(done_ids)}）", flush=True)
    if not pending:
        print("全部完成，无需生成", flush=True)
        return

    ok, err = 0, 0
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(gen_one, d): d for d in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            ok += rec["status"] == "ok"
            err += rec["status"] != "ok"
            if i % 50 == 0 or i == len(futs):
                print(f"进度 {i}/{len(futs)}: ok={ok} err={err}", flush=True)

    print(f"完成: 成功 {ok}, 失败/空 {err}", flush=True)


if __name__ == "__main__":
    main()
