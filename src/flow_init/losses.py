from __future__ import annotations

from typing import Optional
import math

import torch
from torch import nn

from .types import Tensor
from .utils import batch_cholesky


def obs_negloglik(y: Tensor, G: Tensor, x_samples: Tensor, R: Tensor) -> Tensor:
    """
    观测一致性负对数似然（MC）：-log N(y; G x, R)
    y: (B, T, m)
    G: (B, T, m, d) 或 (T, m, d)
    x_samples: (B, T, K, d)
    R: (m, m) 或 (B, T, m, m)
    返回：标量损失
    """
    if R.dim() == 2:
        R = R.expand(y.shape[:-1] + R.shape)
    L = torch.linalg.cholesky(R)
    # 计算残差 r = y - G x
    y_expand = y.unsqueeze(-2)
    y_hat = torch.einsum("btmd,btkd->btkm", G, x_samples)
    r = y_expand - y_hat
    # 求解 L L^T z = r：先解 L u = r，再解 L^T z = u
    # 扩展 L 以对齐 K 维度：(B,T,K,m,m)
    L_exp = L.unsqueeze(2).expand(r.shape[:-1] + L.shape[-2:])
    u = torch.linalg.solve_triangular(L_exp, r.unsqueeze(-1), upper=False).squeeze(-1)
    quad = (u ** 2).sum(dim=-1)  # (B,T,K)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (B,T)
    m = r.shape[-1]
    const = m * math.log(2.0 * math.pi)
    # 将 logdet 从 (B,T) 扩展为 (B,T,K)
    nll = 0.5 * (quad + logdet.unsqueeze(-1) + const)
    return nll.mean()


def prior_kl_mc(x_samples: Tensor, log_q: Optional[Tensor] = None, alpha: float = 1.0) -> Tensor:
    """
    先验 KL（MC）：KL(q || N(0, alpha I)) = E_q[log q - log p0]
    若未提供 log_q，则仅保留 -log p0 项（相当于 L2 正则）。
    x_samples: (B, T, K, d)
    log_q: (B, T, K) 可选
    """
    d = x_samples.shape[-1]
    # log p0
    quad = (x_samples ** 2).sum(dim=-1) / alpha
    logdet = d * torch.log(torch.tensor(alpha, dtype=x_samples.dtype, device=x_samples.device))
    log_p0 = -0.5 * (quad + logdet + d * torch.log(torch.tensor(2.0 * torch.pi, dtype=x_samples.dtype, device=x_samples.device)))
    if log_q is None:
        return (-log_p0).mean()  # drop entropy -> acts as L2 prior
    return (log_q - log_p0).mean()


def dyn_smoothness(x_samples: Tensor, sigma2: float = 1.0) -> Tensor:
    """
    弱动力学一致性项：E ||x_{t+1} - x_t||^2 / sigma2
    x_samples: (B, T, K, d)
    """
    diff = x_samples[:, 1:, ...] - x_samples[:, :-1, ...]
    se = (diff ** 2).sum(dim=-1)
    return (se / sigma2).mean()
