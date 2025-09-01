from __future__ import annotations

from typing import Dict, Optional
from contextlib import nullcontext

import torch

from .encoder import ContextEncoder
from .flows import ConditionalRealNVP
from .features import feat_R, build_token
from .utils import safe_psd_correction
from .config import Config


@torch.no_grad()
def infer_posterior(
    model_cfg: Config,
    encoder: ContextEncoder,
    flow: ConditionalRealNVP,
    y: torch.Tensor,  # (B, T, m) 观测
    G: torch.Tensor,  # (B, T, m, d) 观测矩阵
    R: torch.Tensor,  # (m, m) 或 (B, T, m, m) 噪声协方差
    K: int = 512,
    eps_reg: float = 1e-4,
) -> Dict[str, torch.Tensor]:
    p = next(encoder.parameters())
    device = p.device
    dtype = p.dtype
    y = y.to(device=device, dtype=dtype)
    G = G.to(device=device, dtype=dtype)
    R_feat = feat_R(R if R.dim() == 2 else R[0, 0], mode="diag").to(device=device, dtype=dtype)
    if R.dim() == 2:
        R_feat = R_feat.expand(y.shape[:-1] + (R_feat.shape[-1],))
    else:
        R_feat = torch.diagonal(R, dim1=-2, dim2=-1)
    tokens = build_token(y, G, R_feat, mode_g="vectorize")  # 构建每个 t 的 token 输入
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (getattr(model_cfg, "perf", None) is not None and model_cfg.perf.autocast_bf16 and y.is_cuda)
        else nullcontext()
    )
    with autocast_ctx:
        c = encoder(tokens)  # (B, T, C)
        x_samps = flow.sample(K, c)  # (B, T, K, d) 状态样本
    # 向量化二阶矩：均值/协方差（沿 K 维）
    B, T, K, d = x_samps.shape
    mus = x_samps.mean(dim=2)  # (B,T,d)
    xc = x_samps - mus.unsqueeze(2)  # (B,T,K,d)
    # 协方差：E[(x-μ)(x-μ)^T]
    covs = torch.einsum("btkd,btke->btde", xc, xc) / max(K - 1, 1)
    # PSD 修复（加 epsI）
    eye = torch.eye(d, dtype=dtype, device=device)
    covs = covs + eps_reg * eye.unsqueeze(0).unsqueeze(0)
    return {"mu": mus, "P": covs, "samples": x_samps}


@torch.no_grad()
def calibrate_temperature(
    mu: torch.Tensor,  # (B, T, d)
    P: torch.Tensor,   # (B, T, d, d)
    y: torch.Tensor,   # (B, T, m)
    G: torch.Tensor,   # (B, T, m, d)
    R: torch.Tensor,   # (m, m) or (B, T, m, m)
    grid: torch.Tensor = None,
) -> float:
    device = mu.device
    dtype = mu.dtype
    # 统一 dtype/device，避免 Float/Double 混用
    y = y.to(device=device, dtype=dtype)
    G = G.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    if grid is None:
        grid = torch.logspace(-1, 1.0, 21, device=device, dtype=dtype)  # [0.1, 10]
    best_tau = grid[0]
    best_ll = -torch.inf
    for tau in grid:
        S = torch.einsum("btmd,btdn->btmn", G, tau * P)  # S = G (tau P) G^T + R
        S = S @ G.transpose(-1, -2)
        if R.dim() == 2:
            S = S + R
        else:
            S = S + R
        # 构造残差并计算对数似然
        r = y - torch.einsum("btmd,btd->btm", G, mu)
        L = torch.linalg.cholesky(S)
        z = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
        quad = (r * z).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        ll = (-0.5 * (quad + logdet)).mean()
        if ll > best_ll:
            best_ll = ll
            best_tau = tau
    return float(best_tau.item())


@torch.no_grad()
def state_to_obs_moments(
    mu: torch.Tensor,  # (B, T, d)
    P: torch.Tensor,   # (B, T, d, d)
    G: torch.Tensor,   # (B, T, m, d)
    R: torch.Tensor,   # (m, m) 或 (B, T, m, m)
):
    """计算观测域矩：μ_y,t = G μ_x,t，S_y,t = G P_x,t G^T + R。"""
    device, dtype = mu.device, mu.dtype
    G = G.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    mu_y = torch.einsum("btmd,btd->btm", G, mu)
    GP = torch.einsum("btmd,btdn->btmn", G, P)
    S = GP @ G.transpose(-1, -2)
    if R.dim() == 2:
        S = S + R
    else:
        S = S + R
    return mu_y, S


@torch.no_grad()
def obs_loglik(
    mu: torch.Tensor,  # (B, T, d)
    P: torch.Tensor,   # (B, T, d, d)
    y: torch.Tensor,   # (B, T, m)
    G: torch.Tensor,   # (B, T, m, d)
    R: torch.Tensor,   # (m, m) 或 (B, T, m, m)
    tau: float | None = None,
) -> torch.Tensor:
    """平均观测对数似然（可选温度缩放 P <- tau P），带数值 PSD 修复。"""
    # 统一 dtype/device，避免 Float/Double 混用
    device, dtype = mu.device, mu.dtype
    y = y.to(device=device, dtype=dtype)
    G = G.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    if tau is not None:
        P = tau * P
    mu_y, S = state_to_obs_moments(mu, P, G, R)
    # PSD 修复：优先加 epsI，失败再做特征值截断
    eps = 1e-8
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    S_try = S + eps * eye
    r = y - mu_y
    try:
        L = torch.linalg.cholesky(S_try)
    except Exception:
        w, U = torch.linalg.eigh(S)
        w = torch.clamp(w, min=eps)
        S_fix = U @ torch.diag_embed(w) @ U.transpose(-1, -2)
        L = torch.linalg.cholesky(S_fix)
    z = torch.cholesky_solve(r.unsqueeze(-1), L).squeeze(-1)
    quad = (r * z).sum(dim=-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    m = y.shape[-1]
    const = m * torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))
    ll = -0.5 * (quad + logdet + const)
    return ll.mean()
