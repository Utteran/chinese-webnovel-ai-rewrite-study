# 数据说明与使用边界

仓库包含 `data/dataset.jsonl` 的 1200 段网文原文、`data/dataset_enriched.jsonl` 的 ID/质量字段和初始划分、V1 的 435 对总结重写样本、V2 的 442 对风格模仿样本，以及对应的句法、词汇派生数据。V1 固定划分见 [summary_book_splits.json](../data/summary_book_splits.json)，V2 见 [splits.json](../data/v2/splits.json)。

这些数据含有可能受版权保护的小说原文及衍生文本。仓库未提供逐本作品的来源 URL、授权类型或再分发许可证明。**MIT 许可证只覆盖代码，不自动授权语料。** 继续公开、复用或扩充前，应核对来源和使用许可；未来新增语料应记录采集时间、来源、许可及允许用途。

`scripts/preprocess.py` 现通过 `--source-dir` 接收合法取得的 TXT 语料，输出 `dataset.jsonl`。但已提交的 `dataset_enriched.jsonl` 的完整质量标注步骤未保留，因此从原始 TXT 到现有报告的逐字节全链路复现仍有缺口。V2 先从 enriched 文件的训练部分抽样生成，再按 21 本书建立自己的固定划分。重跑 `train/prepare_v2_splits.py` 默认复用现有映射；`--new-split` 属于新实验，原报告不再对应。

`python scripts/audit_dataset.py` 只读检查配对 ID、标签、书籍划分、句法和词汇派生数据的对齐，以及跨划分完全相同的文本。它不能审核许可，也不能发现语义近重复或作者重叠。
