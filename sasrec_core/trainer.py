from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import pad_sequence
from .model import SASRec


def train_one_epoch(
    model: SASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float | None = 5.0,
) -> float:
    """
    训练一个 epoch。

    损失函数采用逐位置二分类：
    - 正样本：真实下一物品
    - 负样本：随机采样物品
    """
    model.train()
    total_loss = 0.0
    total_steps = 0

    for log_seq, pos_seq, neg_seq in loader:
        log_seq = log_seq.to(device)
        pos_seq = pos_seq.to(device)
        neg_seq = neg_seq.to(device)

        optimizer.zero_grad()

        seq_feats = model(log_seq)
        pos_embs = model.item_embedding(pos_seq)
        neg_embs = model.item_embedding(neg_seq)

        pos_logits = (seq_feats * pos_embs).sum(dim=-1)
        neg_logits = (seq_feats * neg_embs).sum(dim=-1)

        # 仅在有正样本监督的位置计算损失，padding 区域跳过。
        mask = pos_seq > 0
        if mask.sum() == 0:
            continue

        pos_labels = torch.ones_like(pos_logits[mask])
        neg_labels = torch.zeros_like(neg_logits[mask])

        loss = F.binary_cross_entropy_with_logits(pos_logits[mask], pos_labels)
        loss += F.binary_cross_entropy_with_logits(neg_logits[mask], neg_labels)

        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1

    return total_loss / max(1, total_steps)


@torch.no_grad()
def evaluate_ranking(
    model: SASRec,
    user_train_dict: dict[int, list[int]],
    user_valid_dict: dict[int, list[int]],
    user_test_dict: dict[int, list[int]],
    itemnum: int,
    maxlen: int,
    device: torch.device,
    mode: str = "test",
    num_neg: int = 100,
    k: int = 10,
) -> dict[str, float]:
    """
    采样式排名评估，输出 HR@K 与 NDCG@K。

    mode:
    - valid: 用 train 序列预测 valid 目标
    - test : 用 train+valid 序列预测 test 目标
    """
    if mode not in {"valid", "test"}:
        raise ValueError("mode must be 'valid' or 'test'.")

    model.eval()
    hr = 0.0
    ndcg = 0.0
    evaluated_users = 0

    for user_id in user_train_dict.keys():
        train_seq = user_train_dict[user_id]
        if len(train_seq) < 1:
            continue

        if mode == "valid":
            seq = train_seq
            target = user_valid_dict[user_id][0]
        else:
            seq = train_seq + user_valid_dict[user_id]
            target = user_test_dict[user_id][0]

        # 历史交互物品不作为负样本，避免把用户已经喜欢过的物品当负例。
        rated = set(seq)
        rated.add(0)

        negatives = []
        while len(negatives) < num_neg:
            t = np.random.randint(1, itemnum + 1)
            if t not in rated and t != target:
                negatives.append(t)

        candidates = [target] + negatives
        seq_arr = pad_sequence(seq, maxlen)
        seq_tensor = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)
        cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

        # 目标物品位于 candidates[0]，计算它在候选集中的排名。
        scores = model.predict_candidates(seq_tensor, cand_tensor).squeeze(0).cpu().numpy()
        rank = int((scores[1:] > scores[0]).sum()) + 1
        evaluated_users += 1

        if rank <= k:
            hr += 1.0
            ndcg += 1.0 / np.log2(rank + 1)

    return {
        f"HR@{k}": float(hr / max(1, evaluated_users)),
        f"NDCG@{k}": float(ndcg / max(1, evaluated_users)),
        "users": float(evaluated_users),
    }


@torch.no_grad()
def recommend_topk_for_user(
    model: SASRec,
    user_train: dict[int, list[int]],
    user_valid: dict[int, list[int]],
    idx2item: dict[int, int],
    itemnum: int,
    maxlen: int,
    user_idx: int,
    device: torch.device,
    k: int = 10,
    chunk_size: int = 50000,
) -> list[int]:
    """
    为单个用户生成 TopK 推荐（分块打分，适配大候选集）。

    返回值是原始 item_id（通过 idx2item 反查），而不是内部 item_index。
    """
    model.eval()
    seq = user_train[user_idx] + user_valid[user_idx]
    rated = set(seq)

    seq_arr = pad_sequence(seq, maxlen)
    seq_tensor = torch.tensor(seq_arr, dtype=torch.long, device=device).unsqueeze(0)

    kept_scores = []
    kept_items = []
    rated_tensor = torch.tensor(list(rated), dtype=torch.long, device=device) if rated else None

    # 对全量物品按块打分，避免一次性构造超大张量导致显存/内存压力。
    for start in range(1, itemnum + 1, chunk_size):
        end = min(itemnum, start + chunk_size - 1)
        candidates = torch.arange(start, end + 1, dtype=torch.long, device=device)
        scores = model.predict_candidates(seq_tensor, candidates).squeeze(0)

        if rated_tensor is not None and rated_tensor.numel() > 0:
            # 过滤用户历史已交互物品，避免重复推荐。
            mask = torch.isin(candidates, rated_tensor)
            scores = scores.masked_fill(mask, -1e9)

        topn = min(k, scores.numel())
        part_scores, part_idx = torch.topk(scores, k=topn)
        part_items = candidates[part_idx]
        kept_scores.append(part_scores.cpu())
        kept_items.append(part_items.cpu())

    merged_scores = torch.cat(kept_scores)
    merged_items = torch.cat(kept_items)
    final_top = torch.topk(merged_scores, k=min(k, merged_scores.numel())).indices
    top_item_idx = merged_items[final_top].numpy().tolist()
    return [idx2item[i] for i in top_item_idx]
