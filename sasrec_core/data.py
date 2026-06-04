from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def random_neq(left: int, right: int, forbidden_set: set[int]) -> int:
    """在 [left, right) 内采样一个不在 forbidden_set 的整数。"""
    t = np.random.randint(left, right)
    while t in forbidden_set:
        t = np.random.randint(left, right)
    return int(t)


def pad_sequence(seq: list[int], maxlen: int) -> np.ndarray:
    """
    将变长行为序列右对齐到固定长度。

    - 输出长度固定为 maxlen
    - 序列不足时左侧补 0（padding）
    - 序列过长时保留最近的 maxlen 个行为
    """
    arr = np.zeros(maxlen, dtype=np.int64)
    clipped = seq[-maxlen:]
    if clipped:
        arr[-len(clipped) :] = clipped
    return arr


def _to_py_list(value: Any) -> list[Any]:
    """将 parquet 读出的多种列表表示统一转换为 Python list。"""
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else [parsed]
    return list(value)


def _resolve_split_schema(cache_dir: Path) -> dict[str, Any]:
    """统一使用 train/valid/test 命名。"""
    unified = {
        "train_path": cache_dir / "train.parquet",
        "valid_path": cache_dir / "valid.parquet",
        "test_path": cache_dir / "test.parquet",
        "user_col": "user_id",
        "train_col": "train_seq",
        "valid_col": "valid_seq",
        "test_col": "test_seq",
        "user_map_path": cache_dir / "user_index_to_id.parquet",
        "item_map_path": cache_dir / "item2idx_mapping.parquet",
        "item_col": "item_id",
        "item_idx_col": "item_idx",
    }

    if all(unified[k].exists() for k in ("train_path", "valid_path", "test_path")):
        return unified
    raise FileNotFoundError(
        "Missing split parquet files. Expected {train,valid,test}.parquet."
    )


def _iter_user_seq_rows(path: Path, user_col: str, seq_col: str, batch_size: int = 65536):
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=[user_col, seq_col], batch_size=batch_size):
        data = batch.to_pydict()
        users = data[user_col]
        seqs = data[seq_col]
        for user, seq in zip(users, seqs):
            py_seq = _to_py_list(seq) if not isinstance(seq, list) else seq
            yield int(user), [int(x) for x in py_seq]


def _load_target_map(path: Path, user_col: str, target_col: str, batch_size: int = 65536) -> dict[int, int]:
    target_map: dict[int, int] = {}
    for user, seq in _iter_user_seq_rows(path, user_col, target_col, batch_size=batch_size):
        target_map[int(user)] = int(seq[0]) if seq else 0
    return target_map


class SASRecTrainDataset(Dataset):
    """
    SASRec 训练集封装。

    对每个用户构造三条等长序列：
    - log_seq: 模型输入序列
    - pos_seq: 下一个真实物品（正样本）
    - neg_seq: 与正样本同位置的随机负样本
    """

    def __init__(self, user_train_dict: dict[int, list[int]], itemnum: int, maxlen: int):
        self.user_train = user_train_dict
        self.users = list(user_train_dict.keys())
        self.itemnum = int(itemnum)
        self.maxlen = int(maxlen)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[idx]
        seq = self.user_train[user]

        # 固定长度数组，0 表示 padding。
        log_seq = np.zeros(self.maxlen, dtype=np.int64)
        pos_seq = np.zeros(self.maxlen, dtype=np.int64)
        neg_seq = np.zeros(self.maxlen, dtype=np.int64)

        # 从序列末尾开始，构造“当前位置 -> 下一物品”的监督信号。
        nxt = seq[-1]
        ptr = self.maxlen - 1
        seq_set = set(seq)

        for item in reversed(seq[:-1]):
            log_seq[ptr] = item
            pos_seq[ptr] = nxt
            if nxt != 0:
                # 负样本不允许采到该用户历史中已出现物品，减少假负样本。
                neg_seq[ptr] = random_neq(1, self.itemnum + 1, seq_set)
            nxt = item
            ptr -= 1
            if ptr < 0:
                break

        return (
            torch.tensor(log_seq, dtype=torch.long),
            torch.tensor(pos_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long),
        )


class MemmapSASRecTrainDataset(Dataset):
    """
    低内存训练数据集：通过 memmap 按 offset 读取用户序列。

    不在内存中保留完整 user->seq 字典，适用于百万级用户训练。
    """

    def __init__(self, memmap_dir: str | Path, itemnum: int, maxlen: int):
        self.memmap_dir = Path(memmap_dir)
        arrays = load_memmap_arrays(self.memmap_dir)
        self.items = arrays["items"]
        self.offsets = arrays["offsets"]
        self.itemnum = int(itemnum)
        self.maxlen = int(maxlen)

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.offsets[idx])
        end = int(self.offsets[idx + 1])
        seq = np.asarray(self.items[start:end], dtype=np.int64)
        if seq.size == 0:
            raise ValueError(f"Empty training sequence at index={idx}.")

        log_seq = np.zeros(self.maxlen, dtype=np.int64)
        pos_seq = np.zeros(self.maxlen, dtype=np.int64)
        neg_seq = np.zeros(self.maxlen, dtype=np.int64)

        nxt = int(seq[-1])
        ptr = self.maxlen - 1
        seq_set = set(int(x) for x in seq.tolist())

        for item in reversed(seq[:-1]):
            item_int = int(item)
            log_seq[ptr] = item_int
            pos_seq[ptr] = nxt
            if nxt != 0:
                neg_seq[ptr] = random_neq(1, self.itemnum + 1, seq_set)
            nxt = item_int
            ptr -= 1
            if ptr < 0:
                break

        return (
            torch.tensor(log_seq, dtype=torch.long),
            torch.tensor(pos_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long),
        )


def save_split_cache(
    cache_dir: str | Path,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    user_test: dict[int, list[int]],
    user_index_to_id: dict[int, int] | None = None,
    item2idx: dict[int, int] | None = None,
) -> None:
    """
    将 train/valid/test 切分结果写入 parquet 缓存目录。

    可选保存用户与物品映射，便于不同 notebook 间复用并保持一致索引空间。
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"user_id": list(user_train.keys()), "train_seq": list(user_train.values())})
    valid_df = pd.DataFrame({"user_id": list(user_valid.keys()), "valid_seq": list(user_valid.values())})
    test_df = pd.DataFrame({"user_id": list(user_test.keys()), "test_seq": list(user_test.values())})

    train_df.to_parquet(cache / "train.parquet", index=False)
    valid_df.to_parquet(cache / "valid.parquet", index=False)
    test_df.to_parquet(cache / "test.parquet", index=False)

    if user_index_to_id is not None:
        user_map_df = pd.DataFrame(
            {"user_index": list(user_index_to_id.keys()), "user_id": list(user_index_to_id.values())}
        )
        user_map_df.to_parquet(cache / "user_index_to_id.parquet", index=False)

    if item2idx is not None:
        item_map_df = pd.DataFrame({"item_id": list(item2idx.keys()), "item_idx": list(item2idx.values())})
        item_map_df.to_parquet(cache / "item2idx_mapping.parquet", index=False)


def build_memmap_cache(
    cache_dir: str | Path,
    memmap_dir: str | Path | None = None,
    overwrite: bool = False,
    batch_size: int = 65536,
) -> Path:
    """
    将 split parquet 构建为 memmap 缓存，降低训练阶段 RAM 峰值。

    生成文件：
    - train_items.npy
    - train_offsets.npy
    - train_users.npy
    - valid_targets.npy
    - test_targets.npy
    - meta.json
    """
    cache = Path(cache_dir)
    schema = _resolve_split_schema(cache)

    target_dir = Path(memmap_dir) if memmap_dir is not None else cache / "memmap_cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    meta_path = target_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        return target_dir

    num_users = 0
    total_items = 0
    max_item = 0
    for _, seq in _iter_user_seq_rows(
        schema["train_path"], schema["user_col"], schema["train_col"], batch_size=batch_size
    ):
        if not seq:
            continue
        num_users += 1
        total_items += len(seq)
        seq_max = max(seq)
        if seq_max > max_item:
            max_item = seq_max

    if num_users == 0:
        raise ValueError("No non-empty train sequences found; cannot build memmap cache.")

    items_mmap = np.lib.format.open_memmap(target_dir / "train_items.npy", mode="w+", dtype=np.int32, shape=(total_items,))
    offsets_mmap = np.lib.format.open_memmap(
        target_dir / "train_offsets.npy", mode="w+", dtype=np.int64, shape=(num_users + 1,)
    )
    users_mmap = np.lib.format.open_memmap(target_dir / "train_users.npy", mode="w+", dtype=np.int64, shape=(num_users,))

    offsets_mmap[0] = 0
    cursor = 0
    row_idx = 0
    for user_id, seq in _iter_user_seq_rows(
        schema["train_path"], schema["user_col"], schema["train_col"], batch_size=batch_size
    ):
        if not seq:
            continue
        users_mmap[row_idx] = int(user_id)
        seq_arr = np.asarray(seq, dtype=np.int32)
        end = cursor + len(seq_arr)
        items_mmap[cursor:end] = seq_arr
        cursor = end
        row_idx += 1
        offsets_mmap[row_idx] = cursor

    valid_map = _load_target_map(
        schema["valid_path"], schema["user_col"], schema["valid_col"], batch_size=batch_size
    )
    test_map = _load_target_map(
        schema["test_path"], schema["user_col"], schema["test_col"], batch_size=batch_size
    )
    valid_targets_mmap = np.lib.format.open_memmap(
        target_dir / "valid_targets.npy", mode="w+", dtype=np.int32, shape=(num_users,)
    )
    test_targets_mmap = np.lib.format.open_memmap(
        target_dir / "test_targets.npy", mode="w+", dtype=np.int32, shape=(num_users,)
    )
    for i in range(num_users):
        uid = int(users_mmap[i])
        valid_targets_mmap[i] = int(valid_map.get(uid, 0))
        test_targets_mmap[i] = int(test_map.get(uid, 0))

    item_map_path = schema["item_map_path"]
    if item_map_path.exists():
        item_table = pq.read_table(item_map_path, columns=[schema["item_idx_col"]])
        itemnum = int(max(item_table[schema["item_idx_col"]].to_pylist()))
    else:
        itemnum = int(max_item)

    del items_mmap, offsets_mmap, users_mmap, valid_targets_mmap, test_targets_mmap

    meta = {
        "version": 1,
        "num_users": int(num_users),
        "num_interactions": int(total_items),
        "itemnum": int(itemnum),
        "train_items": "train_items.npy",
        "train_offsets": "train_offsets.npy",
        "train_users": "train_users.npy",
        "valid_targets": "valid_targets.npy",
        "test_targets": "test_targets.npy",
        "source_cache_dir": str(cache.resolve()),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_dir


def load_memmap_meta(memmap_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(memmap_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing memmap meta file: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_memmap_arrays(memmap_dir: str | Path) -> dict[str, Any]:
    """打开 memmap 缓存数组（只读）并返回句柄。"""
    root = Path(memmap_dir)
    meta = load_memmap_meta(root)
    arrays = {
        "meta": meta,
        "items": np.load(root / meta["train_items"], mmap_mode="r"),
        "offsets": np.load(root / meta["train_offsets"], mmap_mode="r"),
        "users": np.load(root / meta["train_users"], mmap_mode="r"),
        "valid_targets": np.load(root / meta["valid_targets"], mmap_mode="r"),
        "test_targets": np.load(root / meta["test_targets"], mmap_mode="r"),
    }
    return arrays


def load_memmap_eval_dicts(
    memmap_dir: str | Path,
    max_users: int | None = None,
    include_test: bool = True,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """
    从 memmap 中恢复评估字典。

    可通过 max_users 限制评估样本数，避免在内存受限环境下构建超大字典。
    """
    arrays = load_memmap_arrays(memmap_dir)
    items = arrays["items"]
    offsets = arrays["offsets"]
    users = arrays["users"]
    valid_targets = arrays["valid_targets"]
    test_targets = arrays["test_targets"]

    total = int(offsets.shape[0] - 1)
    limit = total if max_users is None else min(total, int(max_users))

    user_train: dict[int, list[int]] = {}
    user_valid: dict[int, list[int]] = {}
    user_test: dict[int, list[int]] = {}

    for i in range(limit):
        uid = int(users[i])
        start = int(offsets[i])
        end = int(offsets[i + 1])
        seq = np.asarray(items[start:end], dtype=np.int64).tolist()
        user_train[uid] = [int(x) for x in seq]
        valid_t = int(valid_targets[i])
        user_valid[uid] = [valid_t] if valid_t > 0 else []
        if include_test:
            test_t = int(test_targets[i])
            user_test[uid] = [test_t] if test_t > 0 else []
        else:
            user_test[uid] = []

    return user_train, user_valid, user_test


def load_split_cache(
    cache_dir: str | Path,
) -> tuple[
    dict[int, list[int]],
    dict[int, list[int]],
    dict[int, list[int]],
    dict[int, int],
    dict[int, int],
    dict[int, int],
    int,
]:
    """
    从缓存目录恢复 SASRec 训练所需字典结构与映射关系。

    返回顺序：
    (user_train, user_valid, user_test, user_index_to_id, item2idx, idx2item, itemnum)
    """
    cache = Path(cache_dir)
    schema = _resolve_split_schema(cache)
    required = [schema["train_path"], schema["valid_path"], schema["test_path"]]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    train_df = pd.read_parquet(required[0])
    valid_df = pd.read_parquet(required[1])
    test_df = pd.read_parquet(required[2])

    user_train = {
        int(u): [int(i) for i in _to_py_list(seq)]
        for u, seq in zip(train_df[schema["user_col"]], train_df[schema["train_col"]])
    }
    user_valid = {
        int(u): [int(i) for i in _to_py_list(seq)]
        for u, seq in zip(valid_df[schema["user_col"]], valid_df[schema["valid_col"]])
    }
    user_test = {
        int(u): [int(i) for i in _to_py_list(seq)]
        for u, seq in zip(test_df[schema["user_col"]], test_df[schema["test_col"]])
    }

    user_map_path = schema["user_map_path"]
    if user_map_path.exists():
        user_map_df = pd.read_parquet(user_map_path)
        user_index_col = "user_index" if "user_index" in user_map_df.columns else schema["user_col"]
        user_index_to_id = {int(u): int(uid) for u, uid in zip(user_map_df[user_index_col], user_map_df["user_id"])}
    else:
        user_index_to_id = {u: u for u in user_train.keys()}

    item_map_path = schema["item_map_path"]
    if item_map_path.exists():
        item_map_df = pd.read_parquet(item_map_path)
        item2idx = {
            int(item): int(idx)
            for item, idx in zip(item_map_df[schema["item_col"]], item_map_df[schema["item_idx_col"]])
        }
        idx2item = {
            int(idx): int(item)
            for item, idx in zip(item_map_df[schema["item_col"]], item_map_df[schema["item_idx_col"]])
        }
        itemnum = int(item_map_df[schema["item_idx_col"]].max())
    else:
        # 兼容无映射文件场景：退化为“内部索引即原 item_id”。
        all_max = 0
        for split_dict in (user_train, user_valid, user_test):
            for seq in split_dict.values():
                if seq:
                    all_max = max(all_max, max(seq))
        itemnum = int(all_max)
        item2idx = {i: i for i in range(1, itemnum + 1)}
        idx2item = {i: i for i in range(1, itemnum + 1)}

    return user_train, user_valid, user_test, user_index_to_id, item2idx, idx2item, itemnum
