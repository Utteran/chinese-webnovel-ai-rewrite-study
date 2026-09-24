# -*- coding: utf-8 -*-
r"""
网文数据集预处理脚本
输入: --source-dir 指定的 TXT 语料目录（gb18030/UTF-8 混合编码）
输出: dataset.jsonl —— 每行一个 200±40 字的干净网文段落

流程: 抽样书 → 智能转码 → 去头部噪音(广告/简介) → 章节识别 → 正文区间裁剪(避开头尾)
      → 句子级清洗 → 200字段贪心组装 → 质量过滤 → 段级去重 → 抽样输出
"""
import os
import re
import json
import random
import hashlib
import argparse

OUT_JSONL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "dataset.jsonl")
OUT_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample_books.txt")  # 抽样书单
NUM_BOOKS = 30                          # 抽样篇数
PARAS_PER_BOOK = 40                     # 每篇抽样段数
SEED = 42
TARGET_LEN, MIN_LEN, MAX_LEN = 200, 160, 280

# 重复副本识别（(1).txt）
DUP_RE = re.compile(r"\(1\)\.txt$")

# 章节标题
CHAP_RE = re.compile(
    r"^\s*(第[\d一二三四五六七八九十百千零两]+[章节回卷部集][\s　]*.*|序章|楔子|番外|尾声|引子|后记|完本感言)"
)

# 头部噪音（广告/群信息/版权声明）
AD_NOISE_RE = re.compile(
    r"(小说群|群号|网盘|pan\.quark|公众号|本作品来自互联网|加入|备用|分享废卢|刺猬猫|"
    r"侵犯作者|24小时内删除|qq\d{5,}|===========)"
)

# 正文尾部噪音（作者的话/求票）
TAIL_NOISE_RE = re.compile(r"^(本章完|ps[:：]|P\.S|求(票|收藏|月票)|感谢|新书|书友群|完本感言)")

URL_RE = re.compile(r"https?://|www\.|\.com|\.cn")


def is_noise_line(line):
    """整行判噪：广告、群信息、URL、纯符号"""
    s = line.strip().strip("\u3000")
    if not s:
        return True
    if s.startswith("="):
        return True
    if AD_NOISE_RE.search(s):
        return True
    if URL_RE.search(s):
        return True
    return False


def split_sentences(text):
    """按句界分句，保留结尾标点"""
    parts = re.split(r"(?<=[。！？…；])", text)
    return [p.strip().strip("\u3000") for p in parts if p.strip()]


def make_paras(sentences):
    """贪心组装句子成 200±40 字段落，边界落在句界。
    封口规则: 长度进入[MIN,MAX]后，若加下一句会超过 TARGET 即封段。
    """
    paras = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) >= MAX_LEN:
            paras.append(cur)
            cur = s
        elif len(cur) >= MIN_LEN and len(cur) + len(s) > TARGET_LEN:
            paras.append(cur)
            cur = s
        else:
            cur += s
    if len(cur) >= 80:
        paras.append(cur)
    return paras


def quality_ok(para):
    """质量过滤：汉字占比、长度、不含垃圾字符"""
    if len(para) < MIN_LEN - 40 or len(para) > MAX_LEN + 80:
        return False
    hanzi = len(re.findall(r"[\u4e00-\u9fff]", para))
    if hanzi / len(para) < 0.7:            # 汉字占比过低视为乱码/混合文本
        return False
    if re.search(r"[@#￥%&*<>{}]", para):
        return False
    return True


def para_hash(para):
    """段级指纹，用于去重"""
    return hashlib.sha1(para.encode("utf-8")).hexdigest()[:16]


def decode_text(raw):
    """智能编码检测: 先 UTF-8(含BOM) 严格解码，失败再 gb18030"""
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace")


def process_book(path):
    """单本书：返回 [(chapter, text), ...]"""
    raw = open(path, "rb").read()
    text = decode_text(raw)
    lines = text.split("\n")

    # 定位正文起点：第一个章节标题
    body_start = None
    for i, ln in enumerate(lines):
        if CHAP_RE.match(ln.strip().strip("\u3000")):
            body_start = i
            break
    if body_start is None:
        return []

    # 正文区间 = 章节起点 ~ 末尾；避开区间首尾各 10%
    body = lines[body_start:]
    n = len(body)
    lo, hi = int(n * 0.10), int(n * 0.90)
    if hi - lo < 200:                       # 太短的书跳过
        return []

    # 逐行清洗 → 句子
    sentences = []
    cur_chap = body[0].strip().strip("\u3000")[:40]
    for i, ln in enumerate(body):
        if not (lo <= i < hi):
            continue
        s = ln.strip().strip("\u3000")
        if not s:
            continue
        if CHAP_RE.match(s):                # 章节标题
            cur_chap = s[:40]
            continue
        if is_noise_line(s) or TAIL_NOISE_RE.match(s):
            continue
        sentences.extend(split_sentences(s))

    # 组装 200 字段落
    paras = []
    for p in make_paras(sentences):
        if quality_ok(p):
            paras.append((cur_chap, p))
    return paras


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="directory containing licensed TXT sources")
    parser.add_argument("--output-jsonl", default=OUT_JSONL)
    parser.add_argument("--output-books", default=OUT_META)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if not os.path.isdir(args.source_dir):
        parser.error(f"source directory does not exist: {args.source_dir}")
    random.seed(args.seed)
    files = sorted(
        f for f in os.listdir(args.source_dir)
        if f.lower().endswith(".txt") and not DUP_RE.search(f)
    )
    if not files:
        parser.error("source directory contains no usable .txt files")
    sampled = random.sample(files, min(NUM_BOOKS, len(files)))

    all_paras = []                          # (book, chapter, para_id, text)
    with open(args.output_books, "w", encoding="utf-8") as meta:
        for fi, fname in enumerate(sampled, 1):
            paras = process_book(os.path.join(args.source_dir, fname))
            book = fname[:-4]
            if len(paras) < PARAS_PER_BOOK:
                meta.write(f"SKIP({len(paras)}): {fname}\n")
                continue
            # 从中间段序随机抽 PARAS_PER_BOOK 段（避开该书首尾段）
            idx = sorted(random.sample(range(len(paras)), PARAS_PER_BOOK))
            for pi, k in enumerate(idx):
                chap, text = paras[k]
                all_paras.append((book, chap, k, text))
            meta.write(f"OK({len(paras)}段,抽{len(idx)}): {fname}\n")

    # 段级去重（跨书）
    seen, dedup = set(), []
    for book, chap, pid, text in all_paras:
        h = para_hash(text)
        if h in seen:
            continue
        seen.add(h)
        dedup.append({"book": book, "chapter": chap, "para_id": pid, "len": len(text), "text": text})

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in dedup:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lens = [len(p["text"]) for p in dedup]
    print(f"抽样书数: {len(sampled)}")
    print(f"产出段落数: {len(dedup)}（去重前 {len(all_paras)}）")
    if lens:
        print(f"长度: min={min(lens)} max={max(lens)} 均值={sum(lens)/len(lens):.0f}")
    print(f"meta 见 {args.output_books}")


if __name__ == "__main__":
    main()
