# sasrec_core 团队协作指南

## 1. 文档目标

本文档面向项目内其他团队成员，帮助你快速理解并稳定使用 `sasrec_core`：

- 这个包解决什么问题
- 代码结构和关键组件职责
- 标准训练/评估/推荐流程
- 百万级数据下的低内存训练方案（memmap）
- 常见问题与协作规范

如果你只想快速跑通，请先看 `README.md`；如果你要维护、扩展、排障，请优先看本指南。

---

## 2. 设计目标与边界

### 2.1 设计目标

- 将原 notebook 里的 SASRec 逻辑抽离为可复用 Python 包
- 提供 sklearn 风格调用入口：`fit / evaluate / recommend / save / load`
- 支持两种训练输入模式：
  - `dict`：开发调试简单
  - `memmap`：大规模训练低内存

### 2.2 非目标（当前版本）

- 不做在线服务化（HTTP/gRPC serving）
- 不做多机分布式训练
- 不做全量候选精排框架（当前以 sampled ranking 为主）

---

## 3. 目录结构与职责

`sasrec_core/` 下核心文件：

- `config.py`
  - `SASRecConfig`：统一管理模型/训练/评估超参数
- `data.py`
  - 数据转换、缓存读写、Dataset 实现
  - 包含 `SASRecTrainDataset`（dict 模式）和 `MemmapSASRecTrainDataset`（低内存）
- `model.py`
  - `SASRec` 模型定义（Embedding + Transformer Encoder）
- `trainer.py`
  - `train_one_epoch`、`evaluate_ranking`、`recommend_topk_for_user`
- `estimator.py`
  - `SASRecEstimator` 高层封装，对外主入口
- `__init__.py`
  - 对外导出 API
- `README.md`
  - 快速上手文档（偏使用）

---

## 4. 数据契约（Data Contract）

### 4.1 基础约定

- 用户索引：`int`
- 物品索引：`int`
- `0` 预留给 padding
- 有效 item 索引范围：`1..itemnum`

### 4.2 dict 模式输入格式

```python
user_train: dict[int, list[int]]
user_valid: dict[int, list[int]]
user_test: dict[int, list[int]]
```

- `user_train[u]`：训练历史序列
- `user_valid[u]`：验证目标（通常长度 1）
- `user_test[u]`：测试目标（通常长度 1）

### 4.3 parquet 缓存命名

统一使用以下命名：

- 统一格式（推荐）：
  - `train.parquet`
  - `valid.parquet`
  - `test.parquet`
  - `item2idx_mapping.parquet`

---

## 5. 训练模式说明

## 5.1 dict 模式（简单、直观）

适合：

- 小规模数据
- 调试模型逻辑
- 快速验证 API

特点：

- 需要将 `user_train/user_valid/user_test` 全量加载为 Python dict
- RAM 占用高，百万用户不推荐

### 5.2 memmap 模式（推荐用于大数据）

适合：

- 百万级用户训练
- 内存受限环境
- 稳定离线训练任务

特点：

- 先通过 `build_memmap_cache` 构建索引缓存
- 训练时按 offset 从磁盘映射读取，避免全量驻留 RAM
- 可通过 `eval_user_limit` 限制评估用户数，进一步降低内存

---

## 6. 标准工作流

### 6.1 离线训练（推荐流程）

1. 准备 split parquet（train/valid/test）
2. 构建 memmap 缓存（首次）
3. `fit(input_mode="memmap")`
4. `evaluate(mode="valid"/"test")`
5. `save()`

### 6.2 预测与推荐

- 若 checkpoint 保存时 `include_data=True`，load 后可直接 `recommend`
- 若为轻量 checkpoint，仍可在外部传入数据做 `evaluate`

---

## 7. 核心 API 速查

### 7.1 构建低内存缓存

```python
from sasrec_core import build_memmap_cache

memmap_dir = build_memmap_cache(cache_dir)
```

### 7.2 训练（memmap）

```python
est.fit(
    input_mode="memmap",
    cache_dir=cache_dir,
    memmap_dir=memmap_dir,
    itemnum=itemnum,
    idx2item=idx2item,
    eval_user_limit=50000,
)
```

### 7.3 评估

```python
valid_metrics = est.evaluate(mode="valid", eval_user_limit=50000)
test_metrics = est.evaluate(mode="test", eval_user_limit=50000)
```

### 7.4 推荐

```python
topk_items = est.recommend(user_idx=123, k=20)
```

> 注意：`recommend` 依赖 `idx2item` 将内部索引映射回原始 item_id。

---

## 8. 性能与内存建议（百万级）

- 优先使用 `input_mode="memmap"`
- 先将 `batch_size` 控制在可稳定运行范围，再逐步上调
- 将 `maxlen` 设为业务可接受的最小值（对内存与算力都敏感）
- 评估时设置 `eval_user_limit`（例如 3 万到 10 万）
- 若 GPU 紧张，同时调小 `hidden_units`、`num_blocks`

---

## 9. 质量与可复现性规范

- 固定随机种子（`SASRecConfig.seed`）
- 关键实验记录：
  - 配置参数
  - 数据版本（cache 路径/日期）
  - 指标输出
  - checkpoint 路径
- 重要改动需附最小冒烟验证：
  - 至少 1 epoch 能跑通
  - `valid/test` 指标可输出
  - `save/load` 可用

---

## 10. 常见问题（团队版）

### Q1: 训练突然占满 RAM？

通常原因：

- 仍在使用 dict 模式
- 在 notebook 中提前把三份序列全量转成 dict

处理建议：

- 改用 memmap 模式
- 训练阶段避免不必要的全量 DataFrame 常驻

### Q2: `recommend` 报映射相关错误？

通常原因：

- 未传入 `idx2item`
- 使用轻量 checkpoint 且缺少必要上下文

处理建议：

- `fit` 时传 `idx2item`
- 或保存时用 `include_data=True`

### Q3: `valid` 评估是否需要 `user_test`？

不需要。  
`mode="valid"` 只依赖 `train+valid`；`mode="test"` 才需要 `user_test`。

---

## 11. 建议的协作方式

- 新成员先完成一次小数据端到端跑通（10~30 分钟）
- 对性能优化类 PR，必须附：
  - 前后内存峰值对比
  - 训练吞吐或时长对比
  - 指标变化范围说明
- 保持 parquet 文件格式和对外 API 稳定，避免随意变更

---

## 12. 下一步可扩展方向

- 批量推荐接口进一步向量化
- 训练 AMP（混合精度）与更系统的 profiler 支持
- 统一实验日志与模型注册（便于跨团队复现实验）

