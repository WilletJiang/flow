from __future__ import annotations

from typing import Literal

import torch

from .types import Tensor


def feat_G(G: Tensor, mode: Literal["vectorize", "gram"] = "vectorize", topk_sv: int = 0) -> Tensor:
    """
    将 G_t (m×d) 摘要为特征向量：
    - vectorize：直接向量化（适用于规模不大）
    - gram：使用 diag(G^T G)，可选再拼接奇异值的前 top-k
    返回：(..., f_g)
    """
    if mode == "vectorize":
        return G.reshape(*G.shape[:-2], -1)
    # gram mode
    gram_diag = torch.sum(G * G, dim=-2)  # diag(G^T G)
    if topk_sv > 0:
        # Compute top-k singular values approximately with svd on CPU if small
        # Keep simple SVD for now; caller can disable for large shapes
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        S_k = torch.nn.functional.pad(S[..., :topk_sv], (0, max(0, topk_sv - S.shape[-1])))
        return torch.cat([gram_diag, S_k], dim=-1)
    return gram_diag


def feat_R(R: Tensor, mode: Literal["vectorize", "diag"] = "diag") -> Tensor:
    if mode == "vectorize":
        return R.reshape(*R.shape[:-2], -1)
    return torch.diagonal(R, dim1=-2, dim2=-1)


def build_token(y_t: Tensor, G_t: Tensor, R_feat: Tensor, mode_g: str = "vectorize") -> Tensor:
    g_feat = feat_G(G_t, mode=mode_g)
    return torch.cat([y_t, g_feat, R_feat], dim=-1)
