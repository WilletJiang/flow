from __future__ import annotations

import numpy as np
import torch


def _ensure_psd(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """确保协方差为（数值上）正定：优先尝试加 eps*I；若仍失败，特征值截断重建。"""
    B, T, m, _ = S.shape
    eye = torch.eye(m, dtype=S.dtype, device=S.device)
    Sj = S + eps * eye
    try:
        torch.linalg.cholesky(Sj)
        return Sj
    except Exception:
        # eigen 修复
        w, U = torch.linalg.eigh(S)
        w_clamped = torch.clamp(w, min=eps)
        return (U @ torch.diag_embed(w_clamped) @ U.transpose(-1, -2)).to(S.dtype)


def _chi2_icdf(levels: torch.Tensor, df: int, iters: int = 64) -> torch.Tensor:
    """卡方分布反CDF（分位数），使用二分法近似：
    F(x; k) = gammainc(k/2, x/2)。返回与 levels 同形状的阈值。"""
    device = levels.device
    dtype = levels.dtype
    k2 = torch.tensor(df / 2.0, dtype=dtype, device=device)
    # 初始上下界
    lo = torch.zeros_like(levels)
    # 粗略上界：k + 20*sqrt(2k) + 50（足够大）
    up = torch.full_like(levels, df + 20.0 * np.sqrt(2.0 * df) + 50.0)

    def cdf(x):
        return torch.special.gammainc(k2, x * 0.5)

    # 防止 levels 为 0 或 1 的边界
    lv = torch.clamp(levels, 1e-9, 1 - 1e-9)
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        val = cdf(mid)
        mask = val < lv
        lo = torch.where(mask, mid, lo)
        up = torch.where(mask, up, mid)
    return up


@torch.no_grad()
def whiten_residual(y: torch.Tensor, mu_y: torch.Tensor, S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """白化观测残差：z = L^{-1} (y - mu_y)，其中 LL^T ≈ S（数值修复）。"""
    r = y - mu_y
    S_safe = _ensure_psd(S, eps=eps)
    L = torch.linalg.cholesky(S_safe)
    z = torch.linalg.solve_triangular(L, r.unsqueeze(-1), upper=False).squeeze(-1)
    return z


@torch.no_grad()
def coverage_curve(y: torch.Tensor, mu_y: torch.Tensor, S: torch.Tensor, levels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """名义覆盖 vs 经验覆盖：使用马氏距离阈值（卡方分布分位数）。
    返回 empirical: (len(levels),)"""
    B, T, m = y.shape
    r = y - mu_y
    S_safe = _ensure_psd(S, eps=eps)
    L = torch.linalg.cholesky(S_safe)  # (B,T,m,m)
    z = torch.linalg.solve_triangular(L, r.unsqueeze(-1), upper=False).squeeze(-1)
    m2 = (z**2).sum(dim=-1)  # (B,T)
    thr = _chi2_icdf(levels.to(m2.device), df=m)  # (L,)
    # 广播比较：(B,T) <= (L,) -> (L,B,T)
    cmp = (m2.unsqueeze(0) <= thr.view(-1, 1, 1))
    emp = cmp.float().mean(dim=(1, 2))  # (L,)
    return emp


@torch.no_grad()
def pit_values(y: torch.Tensor, mu_y: torch.Tensor, S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PIT：对白化后的 z ~ N(0,I)，PIT_i = Phi(z_i)。返回 (B,T,m)。"""
    from math import erf, sqrt
    z = whiten_residual(y, mu_y, S, eps=eps)
    # 标准正态 CDF via erf：Phi(z)=0.5*(1+erf(z/sqrt(2)))
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype, device=z.device))))


@torch.no_grad()
def timewise_coverage(z: torch.Tensor, q: float = 0.95) -> torch.Tensor:
    """逐时刻校准：对白化 z，统计 |z_i| < Phi^{-1}((1+q)/2) 的比例，返回 (T,) 平均覆盖。"""
    p = (1.0 + q) / 2.0
    thr = torch.sqrt(torch.tensor(2.0, dtype=z.dtype, device=z.device)) * torch.special.erfinv(torch.tensor(2.0 * p - 1.0, dtype=z.dtype, device=z.device))
    inside = (z.abs() < thr).float()  # (B,T,m)
    return inside.mean(dim=(0, 2))  # (T,)
