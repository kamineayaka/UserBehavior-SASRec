# SASRec 团队复现包

本目录可**单独 clone / 提交**，用于团队复现 SASRec 训练与评估：数据在 `data/`，算法包在 [`sasrec_core/`](sasrec_core/)。

## 文档

| 文档 | 内容 |
|------|------|
| [前置知识与SASRec.md](前置知识与SASRec.md) | 必备基础知识 + SASRec 原理简介 |
| [使用指南.md](使用指南.md) | 环境、数据、训练、排障（按步骤操作） |
| [data/README.md](data/README.md) | 数据文件清单与字段说明 |

## 目录结构

```
SASRec/
├── sasrec_core/          # SASRec 算法包（随本目录提交）
├── data/                 # 训练 parquet（本地复制，不入 Git）
├── notebooks/
│   ├── 01_full_train.ipynb
│   ├── 02_grid_search.ipynb
│   └── 03_goal_check.ipynb
├── results/              # 网格搜索输出
├── scripts/
│   ├── copy_data_from_cache.py
│   ├── sync_sasrec_core.py
│   └── build_notebooks.py
├── 前置知识与SASRec.md
├── 使用指南.md
└── requirements.txt
```

## 克隆本仓库

```bash
git clone https://github.com/kamineayaka/UserBehavior-SASRec.git
cd UserBehavior-SASRec
```

## 5 分钟快速开始

```bash
pip install -r requirements.txt
python scripts/copy_data_from_cache.py   # 若 data/ 尚无 parquet（需上级目录有 SASRec_cache，或手动复制 parquet）
```

在 Jupyter 中打开 `notebooks/01_full_train.ipynb`，从上到下运行。

参考指标（全量用户，HR@10 / NDCG@10）：见 `data/baseline/baseline_sasrec_20260425_130830.json`（约 **0.847 / 0.774**）。
