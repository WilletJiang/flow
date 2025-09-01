from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .types import Tensor


def cholesky_logpdf(residual: Tensor, chol_prec: Tensor) -> Tensor:
    """
    在已知精度矩阵的 Cholesky 分解 L（chol_prec = chol(Sigma^{-1})) 情况下，计算 N(0, Sigma) 的对数密度。
    参数：
      residual: (..., m)
      chol_prec: (..., m, m) 精度矩阵的下三角 Cholesky 因子
    返回：(...,) 维度的对数密度
    """
    m = residual.shape[-1]
    z = torch.triangular_solve(residual.unsqueeze(-1), chol_prec, upper=False)[0].squeeze(-1)
    quad = (z**2).sum(dim=-1)
    logdet_prec = 2.0 * torch.log(torch.diagonal(chol_prec, dim1=-2, dim2=-1)).sum(dim=-1)
    return 0.5 * (logdet_prec - quad - m * torch.log(torch.tensor(2.0 * torch.pi, dtype=residual.dtype, device=residual.device)))


def batch_cholesky(A: Tensor) -> Tensor:
    return torch.linalg.cholesky(A)


def batch_solve_cholesky(L: Tensor, B: Tensor) -> Tensor:
    # 已知 A 的 Cholesky 分解 L L^T = A，求解 A X = B
    return torch.cholesky_solve(B, L)


def safe_psd_correction(S: Tensor, eps: float = 1e-6) -> Tensor:
    # 对协方差进行稳定性修正：加上 eps*I 以确保正定/半正定
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    return S + eps * eye


@dataclass
class WelfordState:
    n: int
    mean: Tensor
    M2: Tensor


def welford_init(d: int, device=None, dtype=None) -> WelfordState:
    # 初始化 Welford 状态（在线均值/协方差）
    return WelfordState(n=0, mean=torch.zeros(d, device=device, dtype=dtype), M2=torch.zeros(d, d, device=device, dtype=dtype))


def welford_update(state: WelfordState, x: Tensor) -> WelfordState:
    # x: (..., d)。将批次展平后逐个样本更新统计量
    x = x.reshape(-1, x.shape[-1])
    n = state.n
    mean = state.mean
    M2 = state.M2
    for xi in x:
        n1 = n + 1
        delta = xi - mean
        mean = mean + delta / n1
        delta2 = xi - mean
        M2 = M2 + torch.outer(delta, delta2)
        n = n1
    return WelfordState(n=n, mean=mean, M2=M2)


def welford_finalize(state: WelfordState, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    # 返回均值与样本协方差；当样本不足时回退为 eps*I
    if state.n < 2:
        d = state.mean.shape[-1]
        return state.mean, eps * torch.eye(d, dtype=state.mean.dtype, device=state.mean.device)
    cov = state.M2 / (state.n - 1)
    return state.mean, cov


def matvec_batched(G: Tensor, x: Tensor) -> Tensor:
    # G: (..., m, d), x: (..., d) 或 (..., K, d)
    if x.dim() == G.dim() - 1:
        return (G @ x.unsqueeze(-1)).squeeze(-1)
    elif x.dim() == G.dim():
        return torch.einsum("...md,...kd->...km", G, x)
    else:
        # x shape (..., K, d) while G (..., m, d)
        return torch.einsum("...md,...kd->...km", G, x)


def ridge_inverse(G: Tensor, y: Tensor, R: Tensor, lam: float) -> Tensor:
    """
    通过 Cholesky 求解岭回归逆：
      x = (G^T R^{-1} G + lam I)^{-1} G^T R^{-1} y
    支持时间/批次维度的批量运算。
    形状：
      G: (..., m, d)
      y: (..., m)
      R: (..., m, m) 或 (m, m)
    返回：x: (..., d)
    """
    # Broadcast R if static
    if R.dim() == 2 and G.dim() >= 3:
        R = R.expand(G.shape[:-2] + R.shape)
    m, d = G.shape[-2], G.shape[-1]
    # 对 R 做 Cholesky，并用于精度方程求解
    L_R = torch.linalg.cholesky(R)
    # 通过 cholesky_solve 分别求 R^{-1}y 与 R^{-1}G
    Ry = torch.cholesky_solve(y.unsqueeze(-1), L_R).squeeze(-1)
    RG = torch.cholesky_solve(G, L_R)
    A = torch.matmul(G.transpose(-1, -2), RG)
    eye_d = torch.eye(d, dtype=G.dtype, device=G.device)
    A = A + lam * eye_d
    L_A = torch.linalg.cholesky(A)
    b = torch.matmul(G.transpose(-1, -2), Ry.unsqueeze(-1)).squeeze(-1)
    x = torch.cholesky_solve(b.unsqueeze(-1), L_A).squeeze(-1)
    return x
