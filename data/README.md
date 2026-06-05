# SASRec 训练数据目录

本目录存放团队复现所需的 **切分后 parquet**（从 [GitHub Release](../数据与模型下载.md) 下载，不入 Git）。

## 必需文件

| 文件 | 说明 | 参考大小 |
|------|------|----------|
| `train.parquet` | 训练序列，列：`user_id`, `train_seq` | ~377 MB |
| `valid.parquet` | 验证目标，列：`user_id`, `valid_seq` | ~9 MB |
| `test.parquet` | 测试目标，列：`user_id`, `test_seq` | ~9 MB |
| `item2idx_mapping.parquet` | 物品映射，列：`item_id`, `item_idx` | ~15 MB |

`train_seq` / `valid_seq` / `test_seq` 为**内部物品索引**列表（`0` 为 padding，物品从 `1` 开始）。

## 可选 / 训练后生成

| 路径 | 说明 |
|------|------|
| `memmap_cache/` | 运行 `01_full_train.ipynb` 时由 `build_memmap_cache` 生成 |
| `sasrec_full_memmap.pt` | 全量训练保存的模型 |
| `baseline/` | 参考指标 JSON（已纳入 Git） |

## 准备数据

见 [数据与模型下载.md](../数据与模型下载.md)。简要步骤：

```bash
python scripts/download_release_assets.py
```

或从 [Latest Release](https://github.com/kamineayaka/UserBehavior-SASRec/releases) 手动下载 4 个 parquet 到本目录。

## 校验

```python
from pathlib import Path
cache = Path(".")
for name in ["train.parquet", "valid.parquet", "test.parquet", "item2idx_mapping.parquet"]:
    p = cache / name
    print(name, "OK" if p.exists() else "MISSING", p.stat().st_size if p.exists() else "")
```
