from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    use_full_context: bool = True  # True: 可看双向上下文；False: 因果（仅过去）
    window: int | None = None      # 窗口半径 h；None 表示全序列
    use_xformers: bool = False     # 可选：用 xFormers 的 memory-efficient attention（CUDA 环境）


@dataclass
class FlowConfig:
    num_coupling_layers: int = 8
    hidden: int = 256
    clamp: float = 8.0  # log-scale 截断范围（放宽以避免方差偏小）
    min_log_scale: float = -9.0    # 额外下界，避免过度收缩


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    epochs_warmup: int = 8
    epochs: int = 80
    batch_size: int = 8
    K_train: int = 16
    beta_prior: float = 1.0
    lambda_dyn: float = 0.1
    gamma_entropy: float = 0.02  # 略增以防方差塌缩
    ridge_lambda: float = 1e-2
    entropy_subsample: float = 1.0  # [0,1] 计算熵/日志密度的子采样比例（<1 减少开销）


@dataclass
class PerfConfig:
    autocast_bf16: bool = True     # CUDA 上使用 bf16 autocast
    compile: bool = True           # 启用 torch.compile(inductor)


@dataclass
class InferenceConfig:
    K: int = 512
    eps_reg: float = 1e-4


@dataclass
class ModelConfig:
    state_dim: int = 16
    obs_dim: int = 8
    feature_mode: str = "vectorize"  # or "gram"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)
    perf: PerfConfig = field(default_factory=PerfConfig)
