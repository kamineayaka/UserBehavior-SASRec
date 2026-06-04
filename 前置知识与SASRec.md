# 前置必备知识与 SASRec 简介

本文面向需要在本项目中跑通 SASRec 的团队成员：前半部分说明**需要提前掌握的基础**，后半部分说明 **SASRec 在做什么** 以及与本仓库实现的对应关系。

更完整的实现细节见：[`sasrec_core/SASREC_原理与实现.md`](sasrec_core/SASREC_原理与实现.md)、[`sasrec_core/TEAM_GUIDE.md`](sasrec_core/TEAM_GUIDE.md)。

---

## 第一部分：前置必备知识

### 1. Python 与虚拟环境

- 建议使用 Python 3.10+，用 `venv` 或 Conda 隔离依赖。
- 在 `SASRec/` 目录安装：`pip install -r requirements.txt`
- GPU 训练需单独安装带 CUDA 的 PyTorch（见 [PyTorch 官网](https://pytorch.org/get-started/locally/)）。

### 2. 常用数据科学库

| 库 | 在本项目中的用途 |
|----|------------------|
| **NumPy** | 序列数组、memmap 缓存 |
| **Pandas** | 读取 parquet、查看训练日志 |
| **PyArrow** | parquet 底层读写 |

训练数据为 `data/*.parquet`，列含义见 [data/README.md](data/README.md)。

### 3. PyTorch 最小概念

- **Tensor**：模型输入输出；`device="cuda"` 表示用 GPU。
- **DataLoader**：按 batch 取训练样本。
- **state_dict**：模型权重；`SASRecEstimator.save/load` 会持久化。

你不需要手写反向传播，但要理解：**batch_size、epoch、学习率** 会影响训练时间与显存。

### 4. 推荐系统基础

- **协同过滤（CF）**：根据用户/物品共现做相似推荐。
- **流行度（Most Popular）**：推荐全局最热物品，是常用弱基线。
- **序列推荐**：不仅看“买过什么”，还看**先后顺序**，预测“下一步会交互什么”。

### 5. 评估指标：HR@K 与 NDCG@K

本项目在验证/测试时使用**采样排序评估**（1 个正样本 + 若干负样本）：

- **HR@K（Hit Rate）**：真实下一物品是否出现在 Top-K 中。
- **NDCG@K**：若命中，排名越靠前得分越高。

两者越高越好。评估存在随机负采样波动，固定 `seed` 可提升可复现性。

### 6. 数据切分直觉（Leave-one-out 风格）

每个用户通常有：

- **train**：历史行为序列（用于学习）
- **valid**：验证目标（下一跳）
- **test**：测试目标（在 valid 之后的下一跳）

本仓库的 parquet 已做好切分，训练时无需再从原始日志重切。

---

## 第二部分：SASRec 知识

### 1. 要解决什么问题？

给定用户按时间排序的交互序列  
\(S_u = [i_1, i_2, \ldots, i_T]\)，  
预测**下一个物品** \(i_{T+1}\)。

SASRec（*Self-Attentive Sequential Recommendation*）用 **Transformer 自注意力** 编码历史，使每个位置都能关注序列中任意历史位置（在工程上通过 `maxlen` 截断控制成本）。

### 2. 训练时在学什么？

对每个时间步，用**当前前缀**预测**下一物品**：

- **正样本**：真实的下一物品 embedding
- **负样本**：从全物品空间随机采样（尽量不采用户历史中已出现的物品）

模型输出与正负样本 embedding 的**点积分数**，用二分类交叉熵优化（详见 `sasrec_core/model.py` 与 `trainer.py`）。

### 3. 与本项目 `sasrec_core` 的对应

| 概念 | 代码位置 |
|------|----------|
| 超参数 | `SASRecConfig`（`config.py`） |
| 模型结构 | `SASRec`（`model.py`） |
| 数据 / memmap | `data.py` |
| 训练与评估 | `trainer.py` |
| 一站式 API | `SASRecEstimator`（`estimator.py`） |

大规模训练推荐使用 **memmap 模式**（`build_memmap_cache` + `fit(input_mode="memmap")`），避免把全量 `user_train` 字典载入内存。

### 4. 本项目的参考效果（Baseline）

在约 **98.6 万用户** 的全量切分数据上，冻结基线（见 `data/baseline/baseline_sasrec_20260425_130830.json`）：

| 集合 | HR@10 | NDCG@10 |
|------|-------|---------|
| Valid | ≈ 0.8476 | ≈ 0.7747 |
| Test  | ≈ 0.8475 | ≈ 0.7736 |

复现时因硬件、随机种子、训练轮次不同，指标会有小幅波动；显著偏低时请查 [使用指南.md](使用指南.md) 的排障章节。

### 5. 建议学习路径

1. 阅读本文 → [使用指南.md](使用指南.md) 跑通 `01_full_train.ipynb`
2. 需要调参 → `02_grid_search.ipynb`
3. 需要理解代码与数据流 → [`sasrec_core/SASREC_原理与实现.md`](sasrec_core/SASREC_原理与实现.md)
4. 需要维护/扩展包 → [`sasrec_core/TEAM_GUIDE.md`](sasrec_core/TEAM_GUIDE.md)
