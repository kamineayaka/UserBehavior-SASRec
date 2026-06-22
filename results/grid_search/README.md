# 网格搜索结果目录

团队成员将 FAST 快筛输出提交到此目录。完整任务说明与提交步骤见 [docs/团队协作.md](../../docs/团队协作.md)。

## 文件命名

```
grid_search_fast_{姓名}_{任务ID}_{YYYYMMDD_HHMMSS}.csv
grid_search_fast_{姓名}_{任务ID}_{YYYYMMDD_HHMMSS}.jsonl
```

任务 ID 如 `A1`、`B3`。notebook 默认生成带时间戳的文件名，提交前请重命名。每次同时提交 **csv** 与 **jsonl**。

## 目录说明

| 路径 | 说明 |
|------|------|
| `history/` | 历史 fast 结果（对照用，勿修改） |
| `grid_search_fast_*.csv` | 团队成员提交的结果 |
| `maxlen100_team_B_ranking.md` | B 组排名汇总 |

## 维护者汇总

```bash
python scripts/merge_grid_results.py
```

选定最优超参后运行 `notebooks/04_full_train_B7.ipynb` 做全量训练。

**注意**：FAST 结果为 10 万用户抽样评估，不可与 `data/baseline/*.json` 全量指标直接比大小。
