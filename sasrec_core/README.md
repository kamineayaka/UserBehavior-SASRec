# sasrec_core

从原始 SASRec Notebook 抽离的可复用算法包，提供 sklearn 风格的 API：

| 方法 | 说明 |
|------|------|
| `fit` | 训练（支持 dict / memmap 两种输入模式） |
| `evaluate` | 评估 HR@K / NDCG@K |
| `recommend` | 单用户 TopK 推荐 |
| `save` / `load` | 模型持久化 |

原理与代码对照见 [SASREC_原理与实现.md](SASREC_原理与实现.md)。项目级使用流程见 [docs/使用指南.md](../docs/使用指南.md)。

## 模块结构

| 文件 | 职责 |
|------|------|
| `config.py` | `SASRecConfig` 超参数 |
| `data.py` | 数据集、padding、负采样、memmap 缓存 |
| `model.py` | `SASRec` 网络定义 |
| `trainer.py` | 训练与评估 |
| `estimator.py` | `SASRecEstimator` 高层封装 |

## 快速开始（memmap 模式，推荐）

```python
from pathlib import Path
import torch
from sasrec_core import SASRecConfig, SASRecEstimator, build_memmap_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = Path("./data")

memmap_dir = build_memmap_cache(cache_dir)

config = SASRecConfig(num_epochs=3, batch_size=256, maxlen=50)
est = SASRecEstimator(config=config, device=DEVICE)
est.fit(
    input_mode="memmap",
    cache_dir=cache_dir,
    memmap_dir=memmap_dir,
    eval_user_limit=50000,
)

print(est.evaluate(mode="valid"))
print(est.evaluate(mode="test"))
```

> Notebook 会将项目根目录加入 `sys.path`，可直接 `from sasrec_core import ...`。

## 输入约定

- `user_train` / `user_valid` / `user_test`：`dict[int, list[int]]`
- key 为内部 `user_index`，value 为内部物品索引序列
- 索引 `0` 保留给 padding，有效物品索引为 `1..itemnum`
- `recommend(user_idx)` 传入的是内部索引，非原始 `user_id`

## 保存与加载

```python
est.save(Path("./sasrec_model.pt"))
loaded = SASRecEstimator.load(Path("./sasrec_model.pt"), device="cpu")
```

默认轻量保存（不含 train/valid/test 大字典）。需要 load 后直接 `recommend` 时，使用 `include_data=True`。

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `maxlen` | 50 | 序列最大长度 |
| `hidden_units` | 64 | 嵌入维度 |
| `num_blocks` | 2 | Transformer 层数 |
| `num_heads` | 2 | 注意力头数 |
| `dropout_rate` | 0.2 | Dropout |
| `batch_size` | 256 | 批大小 |
| `lr` | 1e-3 | 学习率 |
| `num_epochs` | 3 | 训练轮次 |
| `eval_num_neg` | 100 | 评估负样本数 |
| `eval_k` | 10 | HR@K / NDCG@K 的 K |
| `seed` | 42 | 随机种子 |

## 常见问题

排障见 [docs/使用指南.md](../docs/使用指南.md#8-常见问题)。算法包特有问题：

- **RAM OOM**：改用 `input_mode="memmap"`，设置 `eval_user_limit`
- **推荐用户不匹配**：`recommend` 需要内部 `user_index`，用 `user_index_to_id` 反查
- **`valid` 评估**：只需 `user_train + user_valid`，不需要 `user_test`
