# 测试辅助工具

此目录保留自动化测试依赖的评测指标、固定标注、历史基准和数据准备工具。
临时评测报告、可视化输出和旧版实验脚本已清理。

- `char_cut_metrics.py` / `char_cut_ground_truth.json`：字符框指标与固定标注。
- `compare_pipeline.py`：流水线指标与合成样例。
- `evaluate_megahan_reviewer.py` / `prepare_megahan_assets.py`：复核器评测和数据校验。
- `recommended_backend.json`：历史基准记录，并非当前策略的精度承诺。

模型安装见根目录 [README](../README.md#下载默认模型)。
