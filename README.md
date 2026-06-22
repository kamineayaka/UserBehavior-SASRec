# UserBehavior-SASRec

SASRec 序列推荐算法的团队复现包：数据在 `data/`，算法在 [`sasrec_core/`](sasrec_core/)。可独立 clone，无需依赖上级 monorepo。

## 文档

| 文档 | 内容 |
|------|------|
| [docs/使用指南.md](docs/使用指南.md) | 环境搭建、数据下载、训练流程、排障 |
| [docs/背景知识.md](docs/背景知识.md) | 推荐系统基础 + SASRec 原理简介 |
| [docs/团队协作.md](docs/团队协作.md) | 网格搜索任务、结果提交、维护者流程 |
| [sasrec_core/README.md](sasrec_core/README.md) | 算法包 API 参考 |
| [sasrec_core/SASREC_原理与实现.md](sasrec_core/SASREC_原理与实现.md) | 原理与代码对照（深入阅读） |
| [data/README.md](data/README.md) | 数据文件字段说明 |

## 目录结构

```
UserBehavior-SASRec/
├── docs/                 # 项目文档（入口见上表）
├── sasrec_core/          # SASRec 算法包
├── data/                 # 训练 parquet（本地准备，不入 Git）
├── notebooks/
│   ├── 01_full_train.ipynb
│   ├── 02_grid_search.ipynb
│   ├── 03_goal_check.ipynb
│   └── 04_full_train_B7.ipynb
├── results/grid_search/  # 网格搜索输出
└── scripts/              # 下载、合并等辅助脚本
```

## 快速开始

```bash
git clone https://github.com/kamineayaka/UserBehavior-SASRec.git
cd UserBehavior-SASRec
pip install -r requirements.txt
python scripts/download_release_assets.py
```

在 Jupyter 中打开 `notebooks/01_full_train.ipynb`，从上到下运行。

**参考指标**（全量约 98.6 万用户，HR@10 / NDCG@10）：见 `data/baseline/baseline_sasrec_20260425_130830.json`（约 **0.847 / 0.774**）。

## 维护者

- 首次发布到 GitHub：见 [GITHUB_SETUP.md](GITHUB_SETUP.md)
- 从 monorepo 同步算法包：`python scripts/sync_sasrec_core.py --force`
