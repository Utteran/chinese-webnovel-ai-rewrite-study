# 文味 Detect · AI 文本味检测器

针对**中文网络小说领域**的 AI 生成文本检测工具，提供网页端 + API 双入口。
采用「句式 Transformer + 词汇 XGBoost」双分支架构，在 logit 空间融合，
专门训练于"AI 最大程度模仿真人文风后仍残留的痕迹"。

> ⚠️ 判定结果表示文本与训练集中 AI 重写样本的相似度（AI 参与概率），
> 不等同于绝对真值，短文本（<120 字）结果仅供参考。

## 特性

- **双分支融合**：句式分支（LTP 句法序列 → Transformer）与词汇分支（jieba 157 维词汇特征 → XGBoost）互补，融合后显著优于单分支
- **概率校准**：两个分支输出经 Platt 校准，融合结果具有概率解释
- **段落级判定**：长文自动分段，逐段评分；最终分按段落字数加权平均
- **长度鲁棒性**：文本长度分布经过消融验证，长度特征非决定性捷径（见报告）
- **不确定提示**：短文本或两分支结论冲突时给出"仅供参考"提示

## 项目结构

```
wangwen_dataset/
├── app.py                     # FastAPI 网页后端（检测入口）
├── static/
│   └── index.html             # 网页前端
├── scripts/
│   ├── preprocess.py          # 语料清洗与 ~200 字段落切分
│   ├── generate_rewrites.py   # 1.0 数据：总结后重写生成（DeepSeek）
│   ├── generate_style_pairs.py# 2.0 数据：风格模仿重写生成（并发100/断点续跑）
│   ├── validate_summary_500.py# 1.0 配对质量验证
│   ├── validate_style_pairs.py# 2.0 配对质量过滤（bigram/LCS 复制检测）
│   ├── parse_syntax.py        # LTP 句法解析 → pos_seq / dep_seq
│   ├── extract_lexical_features.py # 词汇特征提取（分组特征 + train-only 词表）
│   ├── calibrate_branches.py  # 分支概率 Platt 校准
│   ├── eval_final_score.py    # 最终得分链路离线评估（长度分组/聚合）
│   ├── opt_fusion_weights.py  # 融合权重优化实验（网格搜索 + bootstrap）
│   ├── eval_syntax_only.py    # 句式分支单独评估（ROC/PR/阈值）
│   ├── smoke_test.py          # API 冒烟测试
│   └── bench_api.py           # API 性能基准
├── train/
│   ├── prepare_transformer_data.py  # 1.0 配对 → 分类数据 + 按书划分
│   ├── prepare_v2_splits.py         # 2.0 按书划分（防作者风格泄漏）
│   ├── train_transformer_500.py     # 1.0 Transformer 训练
│   ├── train_syntax_transformer.py  # 2.0 句式分支训练
│   ├── train_lexical_xgb.py         # 2.0 词汇分支训练
│   ├── train_lexical_ablation.py    # 词汇特征消融（10 组）
│   ├── train_fusion.py              # 融合层权重学习 + 消融对比
│   └── transformer_500_report.json  # 1.0 训练报告
├── data/
│   ├── dataset.jsonl                # 网文段落语料（1200 段）
│   ├── dataset_enriched.jsonl       # 增强版（id/质量/按书划分）
│   ├── pairs_summary_500.jsonl      # 1.0 原始配对
│   ├── valid_pairs_summary_500.jsonl# 1.0 有效配对（435 对）
│   ├── transformer_data.jsonl       # 1.0 分类训练数据
│   ├── sample_books.txt             # 抽样书单
│   └── v2/                          # 2.0 数据（风格模仿重写）
│       ├── raw_pairs_style_500.jsonl      # 500 条原始配对
│       ├── valid_pairs_style_500.jsonl    # 有效配对（442 对）
│       ├── invalid_pairs_style_500.jsonl
│       ├── syntax_pairs_500.jsonl         # 句法解析序列
│       ├── lexical_features_v2.csv        # 词汇特征表
│       └── splits.json                    # 按书划分
├── reports/
│   └── v2/                        # 2.0 全部评估报告（见下方"结果"）
└── requirements.txt
```

> `model/`（约 950MB 模型权重）**不纳入 git 版本管理**，见下方「模型文件」。

## 快速开始

### 1. 环境

Python 3.9+，建议 GPU（句式 Transformer 推理较快）：

```bash
pip install -r requirements.txt
```

### 2. 获取模型

模型权重不在仓库中（体积超 GitHub 限制）。两种方式：

- **自行训练**：按 README 下方「训练流程」逐步执行 `train/` 脚本
- **请求作者分享**：仓库作者处获取 `model/` 目录（约 950MB），解压后目录结构：
  ```
  model/
  ├── ltp_small/                    # LTP 本地模型
  ├── syntax_transformer_v2/        # 句式分支
  ├── lexical_xgb_v2/model.json     # 词汇分支
  └── transformer_classifier_500/   # 1.0 旧版分类器（可选）
  ```

### 3. 启动网页

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>，粘贴文本即可检测。

## 检测 API

```
POST /api/detect
Content-Type: application/json
Body: {"text": "要检测的文本"}

# 限制：非空；≤ 20000 字
```

响应（JSON）：

```jsonc
{
  "overall_score": 0.9186,        // 最终 AI 参与概率（段落字数加权平均）
  "ai_presence": 0.9186,          // 同上（存在概率视角）
  "ai_strength": 0.9186,          // 同上（强度视角）
  "verdict": { "label": "高度疑似 AI", "level": "high" },
  "uncertain": false,
  "notes": [],
  "stats": { "chars": 189, "sentence_count": 6 },
  "segments": [                   // 逐段明细（按 AI 疑似率降序展示）
    {
      "text": "...", "score": 0.93,
      "syntax_score": 0.78, "lexical_score": 0.95,
      "evidence": 0.86, "ai": true, "chars": 189
    }
  ],
  "model": "文味 Detect 2.0 · 双分支校准融合 + 字数加权判定 (442对风格模仿训练)"
}
```

判定档位（对 AI 参与概率）：`<0.35` 偏人类 · `<0.55` 不确定 · `<0.75` 疑似 AI · `>=0.75` 高度疑似 AI。

## 检测链路

```
原始文本
  │  自动分句 → 贪心组装 140~200 字段落
  ▼
┌─────────────────────────────────────────────────┐
│ 每段并行计算两个分支：                            │
│  句式分支: LTP(pos/dep 序列) → Transformer 分支   │
│  词汇分支: jieba 词汇特征(157维) → XGBoost 分支    │
├─────────────────────────────────────────────────┤
│  Platt 校准 → logit 空间线性加权融合 + bias        │
│  fused = sigmoid(w·logit(句式) + (1-w)·logit(词汇))│
├─────────────────────────────────────────────────┤
│  多段聚合: 按段落字数权值加权平均 → 最终分数        │
└─────────────────────────────────────────────────┘
```

- 融合权重 `w_syntax=0.41, bias=-0.05`（经 [weight_opt_report.json](reports/v2/weight_opt_report.json) 细粒度网格搜索 + bootstrap 验证，词汇分支更强因此权重略偏向词汇）
- 短文本（<120 字）或两分支校准分冲突（>0.45）时置 `uncertain` 并提示

## 训练流程

### 1.0 数据（总结后重写，435 对）

1. `scripts/preprocess.py` — 从网文语料清洗抽样 1200 段人类文本（~200 字段落）
2. `scripts/generate_rewrites.py` — 每段「总结大意 → 基于大意重写」，生成 AI 配对（DeepSeek）
3. `scripts/validate_summary_500.py` — 质量验证 → 435 对有效数据
4. `train/prepare_transformer_data.py` — 展开 text+label，按书划分
5. `train/train_transformer_500.py` — GPU 微调分类器

### 2.0 数据（风格模仿重写，442 对）

2.0 改用「最大程度模仿原文风格重写」，让 AI 尽力逼近真人文风后仍残留的痕迹成为可学习特征：

1. `scripts/generate_style_pairs.py` — 随机抽 500 段，提示词要求模仿叙事视角/语气/句式节奏/用词习惯重写；**并发 100** 调用 DeepSeek，断点续跑 + 实时落盘
2. `scripts/validate_style_pairs.py` — 质量过滤 → 442 对有效数据（bigram 覆盖率 >0.55 或 LCS 比例 >0.25 判为复制，剔除套话前缀/长度异常/低汉字占比）

### 2.0 双分支训练

```bash
python train/prepare_v2_splits.py        # 按书划分（防作者风格泄漏）
python scripts/parse_syntax.py           # LTP 解析 → 句法序列
python scripts/extract_lexical_features.py  # 词汇特征（train-only 词表）
python train/train_syntax_transformer.py # 句式分支
python train/train_lexical_xgb.py        # 词汇分支
python train/train_lexical_ablation.py   # 词汇特征消融（可选）
python scripts/calibrate_branches.py     # 分支概率校准
python train/train_fusion.py             # 融合权重学习
python scripts/opt_fusion_weights.py     # 权重优化实验（可选）
python scripts/eval_final_score.py       # 最终评估
```

## 实验结果

### 2.0 双分支消融（442 对，按书划分，校准后）

| 分支 | Test AUC | Accuracy | F1 | 人类误报率 | AI 检出率 |
|---|---:|---:|---:|---:|---:|
| 句式 Transformer（仅句法序列） | 0.829 | 73.8% | 74.9% | 30.2% | 77.9% |
| 词汇 XGBoost（lex+tf+bg，157维） | 0.970 | 91.9% | 92.0% | 9.3% | 93.0% |
| **双分支融合（w_syntax=0.41）** | **0.981** | **91.9%** | **92.0%** | **9.3%** | **93.0%** |

> 词汇分支是主力；句式分支单独较弱（数据量小 + 语法序列抽象度高），
> 但融合后 AUC 与检出率均提升，验证两分支互补假设。

### 2.0 权重优化实验（reports/v2/weight_opt_report.json）

| 方案 | w_syntax | test AUC (95% CI) | test Brier | 人类误报率 |
|---|---:|---|---:|---:|
| 基线（等权） | 0.50 | 0.9796 (0.960–0.993) | 0.0777 | 10.5% |
| 细粒度全局 | **0.41** | **0.9811** (0.963–0.993) | **0.0722** | **9.3%** |
| 长度分桶自适应 | 各桶不同 | 0.9420（过拟合，否） | 0.1064 | 12.8% |

> 权重收益主要在概率校准与误报率，AUC 提升在 bootstrap 区间内不显著。
> 长度分桶自适应因桶内样本过少严重过拟合，已否掉。

### 1.0 旧版结果（435 对，按书划分）

| 指标 | 值 |
|---|---|
| Test AUC | 0.9997 |
| Test Accuracy | 91.45% |
| AI 检出率 | 82.89% |

## 已知限制

- **数据规模小**：仅 442 对、21 本书、单一生成模型（DeepSeek），泛化到其他 AI 模型/领域需扩充数据
- **仿写数据 ≠ 真实 AI 文本**：模型学的是"风格模仿重写痕迹"，对自由生成/续写/润色/人工修改文本的泛化未验证
- **短文本不稳定**：120–180 字段落误报率约 20%（见 `reports/v2/final_score_eval.json`）
- **句式分支偏弱**：LTP 解析噪声 + 训练样本少，句式分支 AUC 仅 0.83，目前作为互补信号使用

## 已知限制的改进方向

1. **扩充数据矩阵**：多 AI 模型（GPT/Claude/通义/Kimi 等）× 多任务（续写/改写/扩写/润色）× 多领域 × 人工修改比例
2. **独立外部测试集**：与训练完全隔离，验证真实泛化能力
3. **长度捷径验证**：构造长度匹配数据集，确认模型不依赖文本长度
4. **句式分支增强**：补充词性/依存类型/从句结构特征，扩大训练数据

## License

MIT
