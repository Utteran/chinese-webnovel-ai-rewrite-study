# 复现说明

## 可以复现什么

已提交的数据和划分可用 Python 标准库检查；已有 JSON、NPZ 报告可直接阅读。从配对数据重新训练需要机器学习依赖、LTP 和预训练 Transformer；网页推理还需要未提交的约 950 MB 模型权重。原始 TXT 到 `dataset_enriched.jsonl` 的完整质量标注步骤未保留，因而无法保证逐字节重建所有材料。

`requirements.txt` 只给出最低版本，没有依赖锁文件；原实验的 Python/CUDA/库精确版本及模型文件哈希也未完整保存。重新训练得到的指标可能不同。请在单独分支或工作副本运行生成与训练脚本，因为它们会改写 `data/`、`reports/v2/` 与 `model/`。

## 只读核验

```bash
python scripts/audit_dataset.py
python -m unittest discover -s tests -v
python -m compileall -q app.py scripts train tests
```

GitHub Actions 在推送和 PR 时运行相同检查。

## 从已有 V2 配对继续

安装 `requirements.txt` 依赖，并先准备 `model/ltp_small/`。仓库没有为缺失权重提供可验证下载地址或哈希。实际执行顺序是：

```bash
python scripts/parse_syntax.py
python train/prepare_v2_splits.py
python scripts/extract_lexical_features.py
python scripts/audit_dataset.py
python train/train_syntax_transformer.py
python train/train_lexical_ablation.py
python train/train_lexical_xgb.py
python scripts/calibrate_branches.py
python train/train_fusion.py
python scripts/opt_fusion_weights.py
python scripts/eval_final_score.py
```

`prepare_v2_splits.py` 读取句法解析结果，故必须排在 `parse_syntax.py` 后；默认复用已提交的划分。`--new-split` 会产生新实验，原报告不再适用。训练脚本使用 `hfl/chinese-roberta-wwm-ext`，需要镜像时可自行设置 `HF_ENDPOINT`。

`train_fusion.py` 使用粗网格并写入 `fusion_report.json`，而已提交的报告记录后续细网格选择的权重 `w=0.41, bias=-0.05`。直接重跑会改变网页使用的参数。新输出应单独保存，并只用验证集选择部署方案。现有 [权重报告](../reports/v2/weight_opt_report.json) 是历史探索记录；修订后的脚本仅依据验证集推荐方案。

## 从新语料开始

仅在拥有使用权的语料上运行：

```bash
python scripts/preprocess.py --source-dir /path/to/licensed-txt
```

这会生成 `dataset.jsonl` 和 `sample_books.txt`，但不会重建原实验 enriched 数据的全部字段。需要为新实验记录这一步，再运行需要 `DEEPSEEK_API_KEY` 的 `scripts/generate_style_pairs.py`；生成会产生外部 API 费用。不要提交密钥或模型权重，也不要把新数据的指标混称为原 V2 结果。

模型齐备时可用 `python -m uvicorn app:app --host 127.0.0.1 --port 8000` 启动演示，或向 `POST /api/detect` 发送 `{"text":"待研究文本"}`。API 的 `score_interpretation` 给出分数语义。
