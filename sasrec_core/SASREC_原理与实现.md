# SASRec：原理与在本项目（`sasrec_core/`）中的实现

本文面向需要理解算法与代码对应关系的读者：前半部分简述 **SASRec 在做什么**，后半部分逐项说明 **`sasrec_core` 如何实现**。

---

## 第一部分：SASRec 要解决什么问题？

### 1.1 问题定义：序列化推荐（Sequential Recommendation）

给定用户 \(u\) 按时间排序的交互序列  
\[
S_u = [i_1, i_2, \ldots, i_T]
\]  
任务是：用 \(S_u\) **预测下一次行为**（或下一物品 \(i_{T+1}\)）。

与“只看当前快照”的传统协同过滤相比，序列模型尝试利用**先后顺序与长期兴趣演化**。

### 1.2 核心直觉：Self-Attention 建模「依赖整条历史」的兴趣表示

与传统 RNN/CNN 不同，Transformer 的自注意力在每个位置都能**直接或权重化地回看**历史中任意位置——更适合捕获长依赖（在工程上再配合截断序列长度 `maxlen` 控制算力）。

SASRec 论文题为 *Self-Attentive Sequential Recommendation*，即：**用序列自注意力把历史编码成表征，再做下一物品相关性预测**。

### 1.3 训练中常见的监督构造（本项目采用的形式）

对每个用户的一条训练序列，常见做法是按时间从左到右（或等价地从右向左构造对齐）构造「当前前缀 → 下一物品」的监督对。  
本项目在 `Dataset` 里对每个用户从末尾向前扫描，对每个位置 \(t\)：

- **输入前缀**（对齐到固定长度、`0` 为 padding）作为 `log_seq`
- **正样本**：真实下一物品 `pos_seq`
- **负样本**：从物品空间中随机采样、且尽量不采用户历史中出现过的物品（减少假负例）

随后在模型里用表征与正负 embedding **点积**得到 logits，用二分类交叉熵训练（详见下文实现）。

---

## 第二部分：`sasrec_core/` 结构与职责一览

```
sasrec_core/
├── __init__.py      # 对外导出 API
├── config.py       # SASRecConfig（超参数）
├── model.py        # SASRec 网络定义
├── data.py         # 数据集、parquet/memmap、padding、负采样
├── trainer.py      # 训练一步、采样评估 HR/NDCG、全量打分推荐
├── estimator.py    # SASRecEstimator：fit/evaluate/recommend/save/load
├── README.md       # 使用说明
└── TEAM_GUIDE.md   # 团队指南
```

**调用关系简述**：

- 用户脚本 / notebook → `SASRecEstimator.fit/evaluate/...`
- `fit` → `DataLoader` + `train_one_epoch` → `SASRec.forward`
- `evaluate` → `evaluate_ranking` → `predict_candidates`

### 2.1 `config.py`：`SASRecConfig`

[`config.py`](config.py) 用 dataclass 集中管理可调参数，主要分为：

- **序列与模型**：`maxlen`、`hidden_units`、`num_blocks`、`num_heads`、`dropout_rate`
- **训练**：`batch_size`、`lr`、`betas`、`weight_decay`、`grad_clip_norm`、`num_epochs`、`num_workers`
- **采样评估**：`eval_num_neg`（每条目标配套的负样本数）、`eval_k`（HR/NDCG 的 \(K\)）
- **`seed`**：贯穿 Python / NumPy / PyTorch，便于复现

`SASRecEstimator` 将这些字段传给 `SASRec` 与 `trainer` 中的对应逻辑。

---

## 第三部分：`model.py` —— SASRec 网络如何实现？

类 `SASRec`（见 `model.py`）包含：

### 3.1 嵌入层

- `item_embedding`: `Embedding(itemnum + 1, hidden_units, padding_idx=0)`  
  - 索引 `0` 专门留给 **padding**（与数据集 padding 对齐）。
- `pos_embedding`: 长度 `maxlen` 的位置嵌入，与时间步对齐。

序列输入为形如 `[B, L]` 的物品 id 张量，`L` 为截断并对齐后的长度。

### 3.2 编码：`log2feats`

步骤概览：

1. 物品 embedding × \(\sqrt{d}\) 缩放（与常见 Transformer 写法一致，稳定梯度尺度）。
2. 加上位置 embedding，`Dropout`。
3. 将 padding 位置置零，避免噪声。
4. **因果注意力掩码**（causal mask）：位置 \(t\) 只能看见 \(\le t\) 的历史。
5. 使用 `nn.TransformerEncoder` 堆叠 `num_blocks` 层，每层多头注意力。
6. 最终 `LayerNorm` 输出每个时间步的隐藏向量 `[B, L, H]`。

### 3.3 预测：`predict_candidates`

训练与评估里都关心「如何用当前表征给候选打分」：

- 取 **最后一个有效时间步** 的表征 `feats[:, -1, :]` 作为当前用户兴趣向量（因序列右对齐、padding 在左侧，最后一位通常对应最近一次可见行为之后的预测侧表征；具体与你的 `pad_sequence` 约定一致）。
- 候选物品若为共享候选 `[C]`，则与 embedding 做点积得到 `[B, C]` 分数。

---

## 第四部分：`data.py` —— 数据如何从 parquet 到 Tensor？

### 4.1 索引约定（非常重要）

- 用户、物品均以 **整数 id** 表示。
- **`0` 保留给 padding**。有效物品索引范围为 **`1 .. itemnum`**。

### 4.2 `pad_sequence(seq, maxlen)`

- 超长序列：**只保留最近 `maxlen` 个**（右对齐）。
- 不足长度：左侧用 `0` 填充。

### 4.3 `SASRecTrainDataset` / `MemmapSASRecTrainDataset`

两者逻辑一致：**每个样本对一个用户**，输出三元组 `(log_seq, pos_seq, neg_seq)`：

- 从序列尾部向前枚举「当前 token → 下一物品」；
- `neg_seq` 通过 `random_neq` 在 `[1, itemnum]` 区间采样且避开用户历史中已出现集合（近似减少假负例）。

区别在于数据来源：

| 数据集 | 数据从哪来 |
|--------|------------|
| `SASRecTrainDataset` | 内存里的 `dict[user_id -> train_seq]`，适合小规模实验 |
| `MemmapSASRecTrainDataset` | 磁盘 memmap：`train_items.npy` + `train_offsets.npy`，百万级用户时显著省内存 |

### 4.4 缓存：`build_memmap_cache` / parquet 拆分

`_resolve_split_schema` 约定在项目缓存目录中存在：

- `train.parquet` / `valid.parquet` / `test.parquet`
- （可选）`item2idx_mapping.parquet`、`user_index_to_id.parquet`

`build_memmap_cache` 会把训练序列压实到连续数组，并生成 `meta.json`，供 memmap Dataset 读取。

---

## 第五部分：`trainer.py` —— 损失函数与评估如何实现？

### 5.1 训练：`train_one_epoch`

对每个 batch：

- `seq_feats = model(log_seq)` → `[B, L, H]`
- `pos_embs / neg_embs = model.item_embedding(pos_seq/neg_seq)` → `[B, L, H]`
- **逐元素点积**：`pos_logits = (seq_feats * pos_embs).sum(-1)`

损失为 **Masked Binary Cross Entropy**：

- 仅在 `pos_seq > 0` 的位置计算（跳过 padding）。
- `pos_labels = 1`，`neg_labels = 0`。

因此本实现可以理解为：**对每个有效时间步做一次「正物品 vs 负物品」的二元对比**。这与某些论文中使用 softmax sampled softmax 的版本不同，但工程上简单稳定。

### 5.2 离线评估：`evaluate_ranking`

为每个用户构造：

- 序列 `seq`：`valid` 模式下为 `train`；`test` 模式下为 `train + valid`
- **目标物品** \(+\) **`num_neg` 个随机负样本** → 拼接成候选集合
- 用 `predict_candidates` 得到分数，`target` 在 `candidates[0]`
- rank 定义为：**比目标分数更高的负样本个数 + 1**

然后汇总 **HR@K** 与 **NDCG@K**（对每个用户）：

这是一个 **抽样候选集评估**（`1 + num_neg`），不是全量排序——因此成本低，但与真实排序存在 gap；增大 `eval_num_neg` 通常使指标更平滑、更接近难排序情形。

### 5.3 推荐：`recommend_topk_for_user`

对用户历史构建序列后：

- **分块**遍历全部物品 id 区间打分；
- 用 `torch.isin` 把历史中出现过的物品的分数压到极小，避免重复推荐；
- `topk` 合并多块结果，返回 TopK **`item_idx`** 再通过 `idx2item` **映射回原 `item_id`**。

---

## 第六部分：`estimator.py` —— sklearn 风格的封装做了什么？

`SASRecEstimator.fit`：

- **dict 模式**：直接把 `train/valid/test` 字典载入内存数据集。
- **memmap 模式**：从 `cache_dir` 构建或复用 `memmap`，训练时不保留全量字典；评估阶段可按 `eval_user_limit` **抽样用户数**以降低评估内存与时间。

其余方法：

- `evaluate` → 封装 `evaluate_ranking`
- `recommend` → `recommend_topk_for_user`（需要提供 `idx2item`）
- `save/load` → `torch.save` 保存权重与配置上下文

---

## 第七部分：如何阅读源码时的对照表

| 论文/直觉概念 | 代码锚点 |
|---------------|----------|
| 物品embedding | `model.py`：`self.item_embedding` |
| 位置信息与因果遮挡 | `model.py`：`pos_embedding`、`attn_mask` |
| Transformer 编码 | `model.py`：`nn.TransformerEncoder` |
| 最后一跳表征用于打分 | `model.py`：`predict_candidates` 中取 `[:, -1, :]` |
| 前缀 padding 与对齐 | `data.py`：`pad_sequence` |
| Left-to-right（实现上等价从尾构造）的监督对 | `data.py`：`SASRecTrainDataset.__getitem__` |
| BCE + 随机负采样 | `trainer.py`：`train_one_epoch` |
| 采样候选 HR/NDCG | `trainer.py`：`evaluate_ranking` |
| 训练入口 | `estimator.py`：`SASRecEstimator.fit` |

---

## 第八部分：与论文可能存在的实现差异（读代码时应心里有数）

以下为常见差异说明，不作为优劣判断：

1. **损失形式**：本项目使用 **正负各一条 BCE**；论文或参考实现有时是 **正负 logits 组合的 sampled softmax/NCE**。  
2. **评估**：本项目是 **`1 + num_neg` 的采样排名**；若要接近工业全量排序，需要更高成本或离线近似。  
3. **表征使用位置**：训练和评估统一取 **最后一个时间步**；若你希望「中间步」的监督，需要对数据管道与掩码对齐做扩展。

---

## 附录：延伸阅读

- 原始论文：**Kang & McAuley, “Self-Attentive Sequential Recommendation” (IEEE ICDM)**。可在公开渠道检索题名与 bibtex。
- 本仓库快速上手：**[README.md](README.md)**。
