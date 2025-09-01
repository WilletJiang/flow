from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

class Conditioner(nn.Module):
    """子网：输入为被掩蔽的 x_id 与上下文 c，输出对应通过通道的 shift 与 log_scale。"""
    def __init__(self, in_dim: int, context_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x_id: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_id, context], dim=-1))


class AffineCoupling(nn.Module):
    def __init__(self, d: int, context_dim: int, hidden: int, mask: torch.Tensor, clamp: float, min_log_scale: float):
        super().__init__()
        self.register_buffer("mask", mask.to(dtype=torch.bool))  # (d,)
        self.clamp = float(clamp)
        self.min_log_scale = float(min_log_scale)
        d_in = int(mask.to(torch.int).sum().item())
        d_out = d - d_in
        self.cond = Conditioner(d_in, context_dim, hidden, 2 * d_out)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (..., d), context: (..., c)
        m = self.mask
        x_id = x[..., m]  # (..., d_in)
        params = self.cond(x_id, context)
        shift, log_scale = torch.chunk(params, 2, dim=-1)
        log_scale = torch.clamp(log_scale, min=self.min_log_scale, max=self.clamp)
        scale = torch.exp(log_scale)
        x_pass = x[..., (~m)]
        y_pass = x_pass * scale + shift
        # 组装 y
        y = x.clone()
        y[..., (~m)] = y_pass
        log_det = log_scale.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.mask
        y_id = y[..., m]
        params = self.cond(y_id, context)
        shift, log_scale = torch.chunk(params, 2, dim=-1)
        log_scale = torch.clamp(log_scale, min=self.min_log_scale, max=self.clamp)
        scale = torch.exp(log_scale)
        y_pass = y[..., (~m)]
        x_pass = (y_pass - shift) / scale
        x = y.clone()
        x[..., (~m)] = x_pass
        log_det = -log_scale.sum(dim=-1)
        return x, log_det


class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        event_dim: int,
        context_dim: int,
        hidden: int = 256,
        num_layers: int = 8,
        clamp: float = 8.0,
        min_log_scale: float = -9.0,
    ):
        super().__init__()
        self.d = event_dim
        self.c = context_dim
        self.clamp = float(clamp)
        self.min_log_scale = float(min_log_scale)
        # 交替二进制掩码
        masks: List[torch.Tensor] = []
        for i in range(num_layers):
            mask = torch.zeros(event_dim)
            mask[i % 2 :: 2] = 1.0
            masks.append(mask)
        self.register_buffer("masks", torch.stack(masks, dim=0).to(dtype=torch.bool))
        self.layers = nn.ModuleList(
            [
                AffineCoupling(
                    event_dim, context_dim, hidden, self.masks[i], self.clamp, self.min_log_scale
                )
                for i in range(num_layers)
            ]
        )
        # 末端仿射头（岭回归预热/稳定）
        self.head_mu = nn.Sequential(
            nn.Linear(context_dim, hidden), nn.SiLU(), nn.Linear(hidden, event_dim)
        )
        self.head_logscale = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, event_dim),
        )

    def head_params(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.head_mu(context)
        log_scale = torch.clamp(self.head_logscale(context), min=self.min_log_scale, max=self.clamp)
        return mu, log_scale

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.shape[:-1], dtype=z.dtype, device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x, context)
            log_det = log_det + ld
        mu, log_scale = self.head_params(context)
        x = x * torch.exp(log_scale) + mu
        log_det = log_det + log_scale.sum(dim=-1)
        return x, log_det

    def inverse(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_scale = self.head_params(context)
        z = (x - mu) * torch.exp(-log_scale)
        log_det = -log_scale.sum(dim=-1)
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z, context)
            log_det = log_det + ld
        return z, log_det

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z, log_det = self.inverse(x, context)
        const = torch.log(torch.tensor(2.0 * torch.pi, dtype=z.dtype, device=z.device))
        log_base = -0.5 * (z**2 + const).sum(dim=-1)
        return log_base + log_det

    def sample(self, num_samples: int, context: torch.Tensor) -> torch.Tensor:
        shape = context.shape[:-1] + (num_samples, self.d)
        z = torch.randn(shape, dtype=context.dtype, device=context.device)
        ctx = context.unsqueeze(-2).expand(context.shape[:-1] + (num_samples, context.shape[-1]))
        x, _ = self.forward(z, ctx)
        return x
