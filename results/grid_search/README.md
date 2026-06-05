# 网格搜索结果目录

团队成员将 **FAST 快筛** 实验输出提交到此目录，由维护者汇总后选定最优超参，再跑全量训练。

## 文件命名

```
grid_search_fast_{姓名}_{任务ID}_{YYYYMMDD_HHMMSS}.csv
grid_search_fast_{姓名}_{任务ID}_{YYYYMMDD_HHMMSS}.jsonl
```

- **任务 ID**：见 [团队网格搜索任务.md](../../团队网格搜索任务.md)（如 `A1`、`B3`）。
- notebook 默认生成 `grid_search_fast_{时间戳}.csv`，提交前请**重命名**或复制为上述格式。
- 每次任务请同时提交 **csv** 与 **jsonl**。

## 提交方式

1. **Git PR（推荐）**：fork 仓库 → 将文件放入 `results/grid_search/` → 向 `main` 提 PR。
2. **共享盘/群内**：按相同命名打包，由维护者放入本目录并 commit。

## 目录说明

| 路径 | 说明 |
|------|------|
| `history/` | 维护者已有历史 fast 结果（对照用，勿修改） |
| `grid_search_fast_*.csv` | 团队成员新提交的结果 |

## 维护者汇总

```bash
cd SASRec
python scripts/merge_grid_results.py
```

输出排序后的总表，用于选取 Top1 超参跑 `01_full_train.ipynb` 全量训练。

## 注意

- FAST 结果在 **10 万用户** 上评估，指标高于 `data/baseline/*.json` 的全量指标（约 98.6 万用户），**不可直接数值对比**。
- 团队成员**不要**设置 `FAST_MODE=False` 跑全量；全量由维护者完成。
