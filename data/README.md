# SASRec 训练数据目录

本目录存放**切分后 parquet**（从 [GitHub Release](https://github.com/kamineayaka/UserBehavior-SASRec/releases) 下载，不入 Git）。下载步骤见 [docs/使用指南.md](../docs/使用指南.md#3-准备数据)。

## 必需文件

| 文件 | 说明 | 参考大小 |
|------|------|----------|
| `train.parquet` | 训练序列，列：`user_id`, `train_seq` | ~377 MB |
| `valid.parquet` | 验证目标，列：`user_id`, `valid_seq` | ~9 MB |
| `test.parquet` | 测试目标，列：`user_id`, `test_seq` | ~9 MB |
| `item2idx_mapping.parquet` | 物品映射，列：`item_id`, `item_idx` | ~15 MB |

`train_seq` / `valid_seq` / `test_seq` 为**内部物品索引**列表（`0` 为 padding，物品从 `1` 开始）。

## 训练后生成

| 路径 | 说明 |
|------|------|
| `memmap_cache/` | `build_memmap_cache` 生成，可删除后重建 |
| `sasrec_full_memmap.pt` | 全量训练保存的模型 |
| `baseline/` | 参考指标 JSON（入 Git） |

## 校验

```bash
python -c "from pathlib import Path; d=Path('data'); print([p.name for p in sorted(d.glob('*.parquet'))])"
```
