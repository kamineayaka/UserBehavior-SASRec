from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import SASRecConfig
from .data import (
    MemmapSASRecTrainDataset,
    SASRecTrainDataset,
    build_memmap_cache,
    load_memmap_arrays,
    load_memmap_eval_dicts,
    load_memmap_meta,
)
from .model import SASRec
from .trainer import evaluate_ranking, recommend_topk_for_user, train_one_epoch


class SASRecEstimator:
    """
    sklearn 风格的 SASRec 高层封装。

    设计目标：
    1) 在 notebook 中通过 `fit/evaluate/recommend` 快速完成实验闭环
    2) 避免外部代码直接依赖底层训练细节
    3) 支持 save/load，便于跨会话复用模型
    """

    def __init__(self, config: SASRecConfig | None = None, device: str | torch.device | None = None):
        self.config = config or SASRecConfig()
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model: SASRec | None = None
        self.itemnum: int | None = None

        self.user_train: dict[int, list[int]] | None = None
        self.user_valid: dict[int, list[int]] | None = None
        self.user_test: dict[int, list[int]] | None = None
        self.idx2item: dict[int, int] | None = None
        self.user_index_to_id: dict[int, int] | None = None
        self.history: list[dict[str, float]] = []
        self.has_train_data: bool = False
        self.memmap_dir: Path | None = None

        self._seed_everything(self.config.seed)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        """统一设置随机种子，增强实验可复现性。"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _ensure_model(self) -> SASRec:
        """确保模型已初始化（已 fit 或 load）。"""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call fit() or load() first.")
        return self.model

    def _ensure_itemnum(self, itemnum: int | None, user_train: dict[int, list[int]]) -> int:
        """
        确定 itemnum（内部物品索引上界）。

        优先级：
        1) 调用 fit 时显式传入
        2) 对象已保存状态中的 self.itemnum
        3) 从训练字典中扫描最大物品索引
        """
        if itemnum is not None:
            return int(itemnum)
        if self.itemnum is not None:
            return int(self.itemnum)

        max_item = 0
        for seq in user_train.values():
            if seq:
                max_item = max(max_item, max(seq))
        return int(max_item)

    def fit(
        self,
        user_train: dict[int, list[int]] | None = None,
        user_valid: dict[int, list[int]] | None = None,
        user_test: dict[int, list[int]] | None = None,
        itemnum: int | None = None,
        idx2item: dict[int, int] | None = None,
        user_index_to_id: dict[int, int] | None = None,
        verbose: bool = True,
        input_mode: str = "dict",
        cache_dir: str | Path | None = None,
        memmap_dir: str | Path | None = None,
        rebuild_memmap_cache: bool = False,
        eval_user_limit: int | None = 50000,
    ) -> "SASRecEstimator":
        """
        训练 SASRec 模型。

        参数约定：
        - user_train / user_valid / user_test 的 key 为内部用户索引
        - 序列中的 item 为内部物品索引（1..itemnum，0 预留给 padding）
        """
        if input_mode not in {"dict", "memmap"}:
            raise ValueError("input_mode must be 'dict' or 'memmap'.")

        eval_train = user_train
        eval_valid = user_valid
        eval_test = user_test
        if input_mode == "dict":
            if user_train is None or user_valid is None:
                raise ValueError("dict mode requires user_train and user_valid.")
            self.user_train = user_train
            self.user_valid = user_valid
            self.user_test = user_test
            self.user_index_to_id = user_index_to_id
            self.has_train_data = True
            self.memmap_dir = None

            self.itemnum = self._ensure_itemnum(itemnum, user_train)
            self.idx2item = idx2item or {i: i for i in range(1, self.itemnum + 1)}
            dataset = SASRecTrainDataset(user_train, itemnum=self.itemnum, maxlen=self.config.maxlen)
        else:
            if cache_dir is None and memmap_dir is None:
                raise ValueError("memmap mode requires cache_dir or memmap_dir.")
            if memmap_dir is None:
                memmap_root = build_memmap_cache(cache_dir=cache_dir, overwrite=rebuild_memmap_cache)
            else:
                memmap_root = Path(memmap_dir)
                if cache_dir is not None:
                    memmap_root = build_memmap_cache(
                        cache_dir=cache_dir,
                        memmap_dir=memmap_root,
                        overwrite=rebuild_memmap_cache,
                    )
                else:
                    _ = load_memmap_meta(memmap_root)

            self.memmap_dir = Path(memmap_root)
            meta = load_memmap_meta(self.memmap_dir)
            self.itemnum = int(itemnum) if itemnum is not None else int(meta["itemnum"])
            self.idx2item = idx2item
            self.user_index_to_id = user_index_to_id
            self.user_train = None
            self.user_valid = None
            self.user_test = None
            self.has_train_data = False
            dataset = MemmapSASRecTrainDataset(
                memmap_dir=self.memmap_dir,
                itemnum=self.itemnum,
                maxlen=self.config.maxlen,
            )
            if eval_user_limit is not None and eval_user_limit > 0:
                eval_train, eval_valid, eval_test = load_memmap_eval_dicts(
                    self.memmap_dir,
                    max_users=eval_user_limit,
                    include_test=True,
                )

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0,
        )

        # 每次 fit 都创建新模型，保证训练过程自洽、不会复用旧权重。
        self.model = SASRec(
            itemnum=self.itemnum,
            maxlen=self.config.maxlen,
            hidden_units=self.config.hidden_units,
            num_blocks=self.config.num_blocks,
            num_heads=self.config.num_heads,
            dropout_rate=self.config.dropout_rate,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
        )

        self.history.clear()
        for epoch in range(1, self.config.num_epochs + 1):
            loss = train_one_epoch(
                self.model,
                loader,
                optimizer,
                self.device,
                grad_clip_norm=self.config.grad_clip_norm,
            )
            # 统一记录训练日志，便于后续分析与可视化。
            log_row: dict[str, float] = {"epoch": float(epoch), "loss": float(loss)}

            if eval_train is not None and eval_valid is not None:
                valid_metrics = evaluate_ranking(
                    self.model,
                    eval_train,
                    eval_valid,
                    eval_test or {},
                    itemnum=self.itemnum,
                    maxlen=self.config.maxlen,
                    device=self.device,
                    mode="valid",
                    num_neg=self.config.eval_num_neg,
                    k=self.config.eval_k,
                )
                log_row.update(valid_metrics)

            self.history.append(log_row)

            if verbose:
                if eval_train is None or eval_valid is None:
                    print(f"Epoch {epoch:02d} | loss={loss:.4f}")
                else:
                    print(
                        f"Epoch {epoch:02d} | loss={loss:.4f} | "
                        f"valid_HR@{self.config.eval_k}={log_row[f'HR@{self.config.eval_k}']:.4f} | "
                        f"valid_NDCG@{self.config.eval_k}={log_row[f'NDCG@{self.config.eval_k}']:.4f}"
                    )

        return self

    def evaluate(
        self,
        user_train: dict[int, list[int]] | None = None,
        user_valid: dict[int, list[int]] | None = None,
        user_test: dict[int, list[int]] | None = None,
        mode: str = "test",
        num_neg: int | None = None,
        k: int | None = None,
        eval_user_limit: int | None = None,
    ) -> dict[str, float]:
        """评估当前模型，默认输出测试集上的 HR/NDCG。"""
        if mode not in {"valid", "test"}:
            raise ValueError("mode must be 'valid' or 'test'.")

        model = self._ensure_model()
        train = user_train or self.user_train
        valid = user_valid or self.user_valid
        test = user_test or self.user_test
        if (train is None or valid is None) and self.memmap_dir is not None:
            train, valid, test = load_memmap_eval_dicts(
                self.memmap_dir,
                max_users=eval_user_limit,
                include_test=(mode == "test"),
            )
        if train is None or valid is None or self.itemnum is None:
            raise ValueError("Missing train/valid inputs for evaluate().")
        if mode == "test" and test is None:
            raise ValueError("mode='test' requires user_test dictionary.")
        # 传给底层函数时，valid 模式也需要占位 test 参数，这里用空字典兜底。
        eval_test = test if test is not None else {}

        return evaluate_ranking(
            model=model,
            user_train_dict=train,
            user_valid_dict=valid,
            user_test_dict=eval_test,
            itemnum=self.itemnum,
            maxlen=self.config.maxlen,
            device=self.device,
            mode=mode,
            num_neg=num_neg or self.config.eval_num_neg,
            k=k or self.config.eval_k,
        )

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        chunk_size: int = 50000,
    ) -> list[int]:
        """为单个用户生成 TopK 推荐结果（返回原始 item_id）。"""
        model = self._ensure_model()
        if self.idx2item is None:
            raise ValueError("idx2item is required for recommend(); pass it to fit(...) first.")
        if self.user_train is None or self.user_valid is None or self.idx2item is None or self.itemnum is None:
            if self.memmap_dir is None:
                raise ValueError(
                    "Missing train/valid or mapping state. "
                    "Call fit() first, or load() from a checkpoint saved with include_data=True."
                )
            arrays = load_memmap_arrays(self.memmap_dir)
            users = np.asarray(arrays["users"])
            matches = np.where(users == int(user_idx))[0]
            if matches.size == 0:
                raise ValueError(f"user_idx={user_idx} not found in memmap cache.")
            pos = int(matches[0])
            start = int(arrays["offsets"][pos])
            end = int(arrays["offsets"][pos + 1])
            seq = np.asarray(arrays["items"][start:end], dtype=np.int64).tolist()
            valid_t = int(arrays["valid_targets"][pos])
            local_train = {int(user_idx): [int(x) for x in seq]}
            local_valid = {int(user_idx): [valid_t] if valid_t > 0 else []}
            return recommend_topk_for_user(
                model=model,
                user_train=local_train,
                user_valid=local_valid,
                idx2item=self.idx2item,
                itemnum=self.itemnum,
                maxlen=self.config.maxlen,
                user_idx=int(user_idx),
                device=self.device,
                k=k,
                chunk_size=chunk_size,
            )

        return recommend_topk_for_user(
            model=model,
            user_train=self.user_train,
            user_valid=self.user_valid,
            idx2item=self.idx2item,
            itemnum=self.itemnum,
            maxlen=self.config.maxlen,
            user_idx=user_idx,
            device=self.device,
            k=k,
            chunk_size=chunk_size,
        )

    def recommend_batch(self, user_ids: list[int], k: int = 10, chunk_size: int = 50000) -> dict[int, list[int]]:
        """批量用户推荐，内部循环调用单用户 recommend。"""
        return {uid: self.recommend(uid, k=k, chunk_size=chunk_size) for uid in user_ids}

    def save(self, path: str | Path, include_data: bool = False) -> None:
        """
        保存模型与推理所需上下文。

        默认仅保存轻量信息（模型参数、配置、映射、训练历史）。
        如需 load 后直接 recommend（不重新传入训练字典），请设置 include_data=True。
        """
        model = self._ensure_model()
        payload: dict[str, Any] = {
            "config": self.config.__dict__,
            "state_dict": model.state_dict(),
            "itemnum": self.itemnum,
            "idx2item": self.idx2item,
            "user_index_to_id": self.user_index_to_id,
            "history": self.history,
            "has_train_data": bool(include_data and self.has_train_data),
            "memmap_dir": str(self.memmap_dir) if self.memmap_dir is not None else None,
        }
        if include_data:
            payload["user_train"] = self.user_train
            payload["user_valid"] = self.user_valid
            payload["user_test"] = self.user_test
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "SASRecEstimator":
        """
        从检查点恢复 estimator。

        注意：这里显式设置 `weights_only=False`，用于加载包含配置与字典状态的完整 payload。
        """
        payload = torch.load(
            Path(path),
            map_location=device if device is not None else "cpu",
            weights_only=False,
        )
        config = SASRecConfig(**payload["config"])
        est = cls(config=config, device=device)

        est.itemnum = int(payload["itemnum"])
        est.idx2item = payload.get("idx2item")
        est.user_index_to_id = payload.get("user_index_to_id")
        est.user_train = payload.get("user_train")
        est.user_valid = payload.get("user_valid")
        est.user_test = payload.get("user_test")
        est.history = payload.get("history", [])
        est.has_train_data = bool(payload.get("has_train_data", est.user_train is not None and est.user_valid is not None))
        saved_memmap_dir = payload.get("memmap_dir")
        est.memmap_dir = Path(saved_memmap_dir) if saved_memmap_dir else None

        est.model = SASRec(
            itemnum=est.itemnum,
            maxlen=est.config.maxlen,
            hidden_units=est.config.hidden_units,
            num_blocks=est.config.num_blocks,
            num_heads=est.config.num_heads,
            dropout_rate=est.config.dropout_rate,
        ).to(est.device)
        est.model.load_state_dict(payload["state_dict"])
        est.model.eval()
        return est
