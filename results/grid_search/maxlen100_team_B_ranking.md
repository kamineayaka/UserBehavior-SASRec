# maxlen=100 团队 B 组网格结果排名

FAST 快筛统一条件：`maxlen=100`, `hidden_units=128`, `FAST_MODE=True`, `num_epochs=3`, `eval_user_limit=100000`, `eval_num_neg=100`, `batch_size=512`（B3/B4 为 256，多为 CPU）。

**排序依据**：`valid_ndcg10` 降序 → `valid_hr10` → `test_ndcg10`。

生成时间：2026-06-04（由 `scripts/` 汇总脚本根据 `grid_search_fast_*_B*.csv` 生成）

---

## 完整排名表

| 排名 | 任务 | 成员 | num_blocks | num_heads | dropout | batch | valid HR@10 | valid NDCG@10 | test HR@10 | test NDCG@10 | 训练(min) | 评估(min) | 结果文件 |
|:----:|:----:|------|:----------:|:---------:|:-------:|:-----:|:-----------:|:-------------:|:----------:|:------------:|:---------:|:---------:|----------|
| 1 | B7 | 王彬哲 | 3 | 4 | 0.15 | 512 | 0.876750 | **0.794450** | 0.878740 | 0.794881 | 40.2 | 6.9 | `grid_search_fast_王彬哲_B7_20260609_095924.csv` |
| 2 | B5 | 丁战胜 | 3 | 2 | 0.15 | 512 | 0.876410 | 0.794256 | 0.878920 | 0.794822 | 68.2 | 10.1 | `grid_search_fast_丁战胜_B5_20260610_210349.csv` |
| 3 | B3 | 徐昊博 | 2 | 4 | 0.15 | 256 | 0.876580 | 0.793865 | 0.878290 | 0.794105 | 508.0 | 9.0 | `grid_search_fast_徐昊博_B3_20260608_230648.csv` |
| 4 | B6 | 王高硕 | 3 | 2 | 0.20 | 512 | 0.875610 | 0.793607 | 0.878070 | 0.794309 | 72.5 | 19.8 | `grid_search_fast_王高硕_B6_20260609_171006.csv` |
| 5 | B8 | 梁硕 | 3 | 4 | 0.20 | 512 | 0.875940 | 0.793605 | 0.878100 | 0.794124 | 53.6 | 8.2 | `grid_search_fast_梁硕_B8_20260610_191256.csv` |
| 6 | B4 | 戴显峦 | 2 | 4 | 0.20 | 256 | 0.875650 | 0.793462 | 0.877090 | 0.792892 | 1237.9 | 8.3 | `grid_search_fast_戴显峦_B4_20260609_213750.csv` |
| 7 | B1 | 张少杰 | 2 | 2 | 0.15 | 512 | 0.873240 | 0.791923 | 0.874160 | 0.791387 | 52.8 | 9.4 | `grid_search_fast_张少杰_B1_20260609_002451.csv` |
| 8 | B2 | 王岩松 | 2 | 2 | 0.20 | 512 | 0.872440 | 0.791257 | 0.873210 | 0.790937 | 51.6 | 10.4 | `grid_search_fast_王岩松_B2_20260608_211943.csv` |

---

## 简要结论

| 维度 | 观察 |
|------|------|
| **最优超参** | `num_blocks=3`, `num_heads=4`, `dropout_rate=0.15`（B7） |
| **次优** | B5（heads=2）与 B7 差距仅 **0.00019** NDCG@10，可视为并列 |
| **blocks** | `num_blocks=3` 整体优于 `2` |
| **dropout** | `0.15` 略优于 `0.20`（同 blocks/heads 时 B5 vs B6、B7 vs B8） |
| **注意** | B3/B4 使用 `batch_size=256`，训练耗时异常长，与 GPU batch=512 结果宜谨慎对比 |

## 全量训练建议（维护者）

在 [`notebooks/04_full_train_B7.ipynb`](../../notebooks/04_full_train_B7.ipynb) 中一键全量训练，或在 `01_full_train.ipynb` 中手动采用 B7 配置：

```python
maxlen = 100
hidden_units = 128
num_blocks = 3
num_heads = 4
dropout_rate = 0.15
batch_size = 512
lr = 1e-3
num_epochs = 8
eval_user_limit = None
eval_num_neg = 200
```

机器 CSV 副本：[`maxlen100_team_B_ranking.csv`](maxlen100_team_B_ranking.csv)
