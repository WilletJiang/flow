from __future__ import annotations

from typing import Optional
from contextlib import nullcontext

import torch
from torch import nn

from .config import EncoderConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        dtype = torch.get_default_dtype()
        pe = torch.zeros(max_len, d_model, dtype=dtype)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=dtype) * (-torch.log(torch.tensor(10000.0, dtype=dtype)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class ContextEncoder(nn.Module):
    def __init__(self, in_dim: int, cfg: EncoderConfig):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model),
            nn.SiLU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.cfg = cfg
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.pe = PositionalEncoding(cfg.d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.out_ln = nn.LayerNorm(cfg.d_model)
        self.use_xf = bool(getattr(cfg, "use_xformers", False))  # 预留开关，不破坏现有路径
        self._xf_available = False
        if self.use_xf:
            try:
                import xformers.ops as xops  # noqa: F401
                self._xf_available = True
            except Exception:
                self.use_xf = False  # 环境不可用则回退

    def _build_attn_mask(self, T: int, device) -> torch.Tensor | None:
        # 根据配置构造注意力 mask（布尔）：True 表示禁止注意力。
        # 使用布尔 mask 有助于命中 PyTorch 2.x 的 SDPA 高效内核。
        if self.cfg.use_full_context and self.cfg.window is None:
            return None
        # 基于窗口的允许集：abs(i-j) <= h
        if self.cfg.window is not None:
            h = int(self.cfg.window)
            idx = torch.arange(T, device=device)
            diff = idx[None, :] - idx[:, None]
            allow_win = (diff.abs() <= h)
        else:
            allow_win = torch.ones((T, T), dtype=torch.bool, device=device)

        if self.cfg.use_full_context:
            # 双向上下文 + （可选）窗口
            mask = ~allow_win
        else:
            # 因果：禁止看未来，再叠加窗口限制
            causal_block = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
            mask = causal_block | (~allow_win)
        return mask

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        # token_seq: (B, T, F)
        # Cast token_seq to the same dtype as the model's weights to prevent dtype mismatch
        token_seq = token_seq.to(self.input_proj[0].weight.dtype)
        h = self.input_proj(token_seq)
        h = self.pe(h)
        attn_mask = self._build_attn_mask(h.size(1), h.device)
        # 为保持权重与行为一致，默认仍使用 PyTorch TransformerEncoder（其内部已走 SDPA 高效核）。
        # xFormers 路径保留为将来替换并迁移权重的选项。
        h = self.encoder(h, mask=attn_mask)
        return self.out_ln(h)  # 输出上下文 c_t: (B, T, d_model)
