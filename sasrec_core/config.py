from dataclasses import dataclass


@dataclass
class SASRecConfig:
    """
    SASRec 训练与推理配置。

    该配置对象尽量保持“算法参数”与“训练流程参数”集中管理，
    便于在 notebook / 脚本中像 sklearn 一样快速创建与复现实验。
    """

    # 序列建模参数
    # 仅保留最近 maxlen 个行为，超长序列会从左侧截断。
    maxlen: int = 50
    # 物品与位置嵌入维度，同时也是 Transformer 隐状态维度。
    hidden_units: int = 64
    # Transformer Encoder 堆叠层数。
    num_blocks: int = 2
    # 多头注意力头数。
    num_heads: int = 2
    # 嵌入与 Transformer 内部 dropout。
    dropout_rate: float = 0.2

    # 训练超参数
    batch_size: int = 256
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.98)
    weight_decay: float = 0.0
    # 梯度裁剪阈值，None 或 <=0 表示关闭。
    grad_clip_norm: float | None = 5.0
    num_epochs: int = 3
    num_workers: int = 0

    # 评估参数（采样评估）
    # 每个目标样本搭配的负样本个数，越大越接近真实排序难度。
    eval_num_neg: int = 100
    # 计算 HR@K / NDCG@K 的 K 值。
    eval_k: int = 10
    # 随机种子，影响负采样与参数初始化等随机过程。
    seed: int = 42
