from __future__ import annotations

import math

import torch
import torch.nn as nn


class SASRec(nn.Module):
    """
    SASRec 模型主体（Self-Attentive Sequential Recommendation）。

    输入：用户历史物品序列（已转为内部索引，0 为 padding）
    输出：每个时间步对应的序列表征，可用于下一物品预测。
    """

    def __init__(
        self,
        itemnum: int,
        maxlen: int,
        hidden_units: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.itemnum = int(itemnum)
        self.maxlen = int(maxlen)
        self.hidden_units = int(hidden_units)

        # item embedding: 物品索引 -> 稠密向量；0 位固定给 padding。
        self.item_embedding = nn.Embedding(self.itemnum + 1, self.hidden_units, padding_idx=0)
        # position embedding: 显式提供序列位置信息。
        self.pos_embedding = nn.Embedding(self.maxlen, self.hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        # 使用 PyTorch 标准 TransformerEncoderLayer。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=num_heads,
            dim_feedforward=self.hidden_units * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(self.hidden_units)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """将行为序列编码为每个时间步的隐藏向量。"""
        # 按论文常见做法对嵌入做 sqrt(d) 缩放，稳定训练初期数值范围。
        seqs = self.item_embedding(log_seqs) * math.sqrt(self.hidden_units)

        positions = torch.arange(log_seqs.size(1), device=log_seqs.device).unsqueeze(0)
        positions = positions.expand_as(log_seqs)
        seqs = seqs + self.pos_embedding(positions)
        seqs = self.dropout(seqs)

        # padding 位不应参与注意力与输出学习。
        timeline_mask = log_seqs.eq(0)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        # 因果掩码：位置 t 只能看见 <= t 的历史，禁止“看未来”。
        attn_mask = torch.triu(
            torch.ones((log_seqs.size(1), log_seqs.size(1)), device=log_seqs.device, dtype=torch.bool),
            diagonal=1,
        )
        feats = self.encoder(seqs, mask=attn_mask, src_key_padding_mask=timeline_mask)
        return self.layer_norm(feats)

    def forward(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """前向等价于 log2feats，便于在训练代码中直接调用 model(x)。"""
        return self.log2feats(log_seqs)

    def predict_candidates(self, log_seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        对候选物品打分。

        取最后一个时间步表示作为用户当前兴趣向量，与候选 item embedding 点积得到相关性分数。

        支持两种候选输入：
        - candidates: [C]，返回 [B, C]
        - candidates: [B, C]，返回 [B, C]
        """
        feats = self.log2feats(log_seq)
        final_feat = feats[:, -1, :]  # [B, H]

        if candidates.dim() == 1:
            # 候选对所有样本共享：[C, H] x [B, H] -> [B, C]
            item_embs = self.item_embedding(candidates)  # [C, H]
            return torch.matmul(final_feat, item_embs.t())

        if candidates.dim() == 2:
            # 每个样本独立候选：[B, C, H] 与 [B, H] 点积 -> [B, C]
            item_embs = self.item_embedding(candidates)
            return torch.sum(item_embs * final_feat.unsqueeze(1), dim=-1)

        raise ValueError("candidates must be 1D [C] or 2D [B, C].")
