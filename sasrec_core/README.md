# sasrec_core 使用说明

## 作用

`sasrec_core` 是从 `SASRec.ipynb` 抽离出的可复用算法核心包，目标是像 sklearn 一样在任意 notebook 中直接调用：

- `fit`：训练
- `evaluate`：评估（HR@K / NDCG@K）
- `recommend`：单用户 TopK 推荐
- `save` / `load`：模型与上下文持久化

---

## 目录结构

- `config.py`：`SASRecConfig` 超参数配置
- `data.py`：数据集、padding、负采样、缓存读写
- `model.py`：SASRec 模型定义
- `trainer.py`：训练与评估函数
- `estimator.py`：高层封装 `SASRecEstimator`
- `__init__.py`：对外导出入口

---

## 环境依赖

最小依赖：

- `torch`
- `numpy`
- `pandas`
- `pyarrow`（读取/写入 parquet 时需要）

安装示例：

```bash
pip install torch numpy pandas pyarrow
```

---

## 快速开始（Notebook，字典模式）

> 下面示例默认你当前工作目录在 `SASRec/` 或 `SASRec/notebooks/` 下。  
> Notebook 会将 `SASREC_DIR` 加入 `sys.path`，可直接 `from sasrec_core import ...`。

```python
from pathlib import Path
import torch

from sasrec_core import SASRecConfig, SASRecEstimator, load_split_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 读取切分缓存
(
    user_train,
    user_valid,
    user_test,
    user_index_to_id,
    item2idx,
    idx2item,
    itemnum,
) = load_split_cache(Path("./data"))

# 配置
config = SASRecConfig(
    maxlen=50,
    hidden_units=64,
    num_blocks=2,
    num_heads=2,
    dropout_rate=0.2,
    batch_size=256,
    lr=1e-3,
    num_epochs=3,
    eval_num_neg=50,
    eval_k=10,
)

# 初始化并训练
est = SASRecEstimator(config=config, device=DEVICE)
est.fit(
    user_train=user_train,
    user_valid=user_valid,
    user_test=user_test,
    itemnum=itemnum,
    idx2item=idx2item,
    user_index_to_id=user_index_to_id,
    verbose=True,
)

# 评估
valid_metrics = est.evaluate(mode="valid")
test_metrics = est.evaluate(mode="test")
print("valid:", valid_metrics)
print("test:", test_metrics)

# 推荐（注意这里传的是内部 user_index，不是原始 user_id）
demo_user = list(user_train.keys())[0]
top10 = est.recommend(user_idx=demo_user, k=10, chunk_size=20000)
print("top10:", top10)
```

---

## 低内存训练（memmap 模式，百万级推荐）

当数据规模很大时，推荐先构建 memmap 缓存，再让 `fit(input_mode="memmap")` 直接按磁盘索引流式取样，避免把完整 `user_train` 全量放入 RAM。

```python
from pathlib import Path
import torch

from sasrec_core import SASRecConfig, SASRecEstimator, build_memmap_cache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = Path("./data")  # 使用 train/valid/test 命名

# 1) 一次性构建低内存缓存（已存在时可复用）
memmap_dir = build_memmap_cache(cache_dir)

# 2) 训练（不传 user_train/user_valid 字典）
config = SASRecConfig(num_epochs=3, batch_size=256, maxlen=50)
est = SASRecEstimator(config=config, device=DEVICE)
est.fit(
    input_mode="memmap",
    cache_dir=cache_dir,
    memmap_dir=memmap_dir,
    rebuild_memmap_cache=False,
    eval_user_limit=50000,  # 每轮仅抽样部分用户评估，进一步节省内存
)

# 3) 评估（可继续走 memmap，按需限制评估用户数）
valid_metrics = est.evaluate(mode="valid", eval_user_limit=50000)
test_metrics = est.evaluate(mode="test", eval_user_limit=50000)
print(valid_metrics, test_metrics)
```

> 说明：如果要 `recommend(...)`，请在 `fit(...)` 时提供 `idx2item`（用于将内部索引反查为原始 item_id）。

---

## 输入数据格式

`fit` 使用的 `user_train` / `user_valid` / `user_test` 需满足：

- 类型：`dict[int, list[int]]`
- key：内部用户索引（`user_index`）
- value：内部物品索引序列（`item_index`）
- 约定：`0` 保留给 padding，物品索引从 `1..itemnum`

示例：

```python
user_train = {
    1: [10, 25, 37, 91],
    2: [5, 8, 13],
}
user_valid = {
    1: [15],
    2: [21],
}
user_test = {
    1: [18],
    2: [34],
}
itemnum = 100
idx2item = {i: i for i in range(1, itemnum + 1)}
```

---

## 保存与加载

```python
from pathlib import Path
from sasrec_core import SASRecEstimator

save_path = Path("./sasrec_model.pt")
# 默认轻量保存（不含 train/valid/test 大字典）
est.save(save_path)

loaded_est = SASRecEstimator.load(save_path, device="cpu")
# 轻量模型可直接评估（若外部传入数据）
metrics = loaded_est.evaluate(
    user_train=user_train,
    user_valid=user_valid,
    user_test=user_test,
    mode="test",
)
print(metrics)

# 如需 load 后直接 recommend，可在保存时 include_data=True
est.save(Path("./sasrec_model_with_data.pt"), include_data=True)
loaded_with_data = SASRecEstimator.load(Path("./sasrec_model_with_data.pt"), device="cpu")
recs = loaded_with_data.recommend(user_idx=1, k=10)
print("top10:", recs)
```

保存内容包含：

- 模型参数（`state_dict`）
- 配置（`SASRecConfig`）
- `itemnum`、`idx2item`、`user_index_to_id`
- 训练日志 `history`
- 可选 `user_train` / `user_valid` / `user_test`（`include_data=True` 时）

---

## 常见问题排查

### 1) `ModuleNotFoundError: No module named 'sasrec_core'`

原因：当前工作目录不在 `SASRec/`，或 Python 路径未包含 `SASREC_DIR`。  
处理：先运行复现包 notebook 的路径 cell，或 `sys.path.insert(0, "<SASRec 绝对路径>")`。

### 2) `FileNotFoundError: Missing file: ...train.parquet`

原因：`data/`（或缓存目录）中缺少切分文件。  
处理：先在数据准备流程中生成 `train.parquet / valid.parquet / test.parquet`，再调用 `load_split_cache()`。

### 3) 训练时显存不足（CUDA OOM）

可优先调小以下参数：

- `batch_size`
- `maxlen`
- `hidden_units`
- `num_blocks`

### 4) 推荐结果用户不匹配

`recommend(user_idx=...)` 需要的是内部 `user_index`，不是原始 `user_id`。  
若需要显示原始用户 ID，请用 `user_index_to_id` 做反查。

### 5) 评估结果波动较大

`evaluate()` 使用了随机负采样，存在统计波动。  
可通过以下方式稳定结果：

- 固定 `seed`
- 提高 `eval_num_neg`
- 多次评估取平均

### 6) `mode='valid'` 是否一定要传 `user_test`？

不需要。`valid` 模式只依赖 `user_train + user_valid`。  
只有 `mode='test'` 才要求 `user_test`。

### 7) RAM OOM（CPU 内存不足）怎么办？

优先改为 `input_mode="memmap"`，并启用以下策略：

- `build_memmap_cache(...)` 后再训练
- `fit(..., eval_user_limit=50000)` 限制每轮评估用户数
- 适当降低 `batch_size` 与 `maxlen`

---

## 建议实践

1. 先用小参数（`num_epochs=1~3`）跑通全流程。
2. 再逐步调大 `maxlen / hidden_units / num_blocks` 做效果优化。
3. 每次实验记录 `config` 与 `history`，保证结果可复现、可比较。
