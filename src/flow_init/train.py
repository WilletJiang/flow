from __future__ import annotations

from typing import Dict, Optional

import math
import torch
from torch import nn
from contextlib import nullcontext
from torch.optim import AdamW

from .config import Config
from .encoder import ContextEncoder
from .flows import ConditionalRealNVP
from .features import feat_R, build_token
from .losses import obs_negloglik, prior_kl_mc, dyn_smoothness
from .utils import ridge_inverse


class PosteriorAmortizer(nn.Module):
    def __init__(self, cfg: Config, in_feat: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = ContextEncoder(in_feat, cfg.model.encoder)
        self.flow = ConditionalRealNVP(
            event_dim=cfg.model.state_dim,
            context_dim=cfg.model.encoder.d_model,
            hidden=cfg.model.flow.hidden,
            num_layers=cfg.model.flow.num_coupling_layers,
            clamp=cfg.model.flow.clamp,
            min_log_scale=cfg.model.flow.min_log_scale,
        )
        # 可选：torch.compile（inductor）。不支持则回退
        if getattr(cfg, "perf", None) is not None and cfg.perf.compile:
            try:
                self.encoder = torch.compile(self.encoder, mode="max-autotune", dynamic=True)
                self.flow = torch.compile(self.flow, mode="max-autotune", dynamic=True)
            except Exception:
                pass

    def forward_samples(self, K: int, tokens: torch.Tensor) -> torch.Tensor:
        c = self.encoder(tokens)
        return self.flow.sample(K, c)


def build_tokens(y: torch.Tensor, G: torch.Tensor, R: torch.Tensor, mode_g: str = "vectorize") -> torch.Tensor:
    # y: (B, T, m), G: (B, T, m, d)
    if R.dim() == 2:
        R_feat = feat_R(R, mode="diag").to(y)
        R_feat = R_feat.expand(y.shape[:-1] + (R_feat.shape[-1],))
    else:
        R_feat = torch.diagonal(R, dim1=-2, dim2=-1)
    return build_token(y, G, R_feat, mode_g=mode_g)


def warmup_epoch(
    model: PosteriorAmortizer,
    batch: Dict[str, torch.Tensor],
    ridge_lambda: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    y, G, R = batch["y"], batch["G"], batch["R"]
    # 统一到模型参数的 dtype/device，避免 Double/Float 混用
    p = next(model.parameters())
    y = y.to(device=p.device, dtype=p.dtype)
    G = G.to(device=p.device, dtype=p.dtype)
    R = R.to(device=p.device, dtype=p.dtype)
    tokens = build_tokens(y, G, R)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (getattr(model.cfg, "perf", None) is not None and model.cfg.perf.autocast_bf16 and y.is_cuda)
        else nullcontext()
    )
    with autocast_ctx:
        c = model.encoder(tokens)
    with torch.no_grad():
        xt = ridge_inverse(G, y, R if R.dim() == 2 else R, ridge_lambda)
    with autocast_ctx:
        mu, _ = model.flow.head_params(c)
    loss = torch.nn.functional.mse_loss(mu, xt.to(dtype=mu.dtype))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.train.grad_clip)
    optimizer.step()
    return float(loss.item())


def training_epoch(
    model: PosteriorAmortizer,
    batch: Dict[str, torch.Tensor],
    K: int,
    beta_prior: float,
    lambda_dyn: float,
    gamma_entropy: float,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    y, G, R = batch["y"], batch["G"], batch["R"]
    p = next(model.parameters())
    y = y.to(device=p.device, dtype=p.dtype)
    G = G.to(device=p.device, dtype=p.dtype)
    R = R.to(device=p.device, dtype=p.dtype)
    tokens = build_tokens(y, G, R)
    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (getattr(model.cfg, "perf", None) is not None and model.cfg.perf.autocast_bf16 and y.is_cuda)
        else nullcontext()
    )
    with autocast_ctx:
        c = model.encoder(tokens)
        x_samps = model.flow.sample(K, c)
    # Log q for entropy (optional)
    B, T, K_, d = x_samps.shape
    log_q: Optional[torch.Tensor]
    p = float(getattr(model.cfg.train, "entropy_subsample", 1.0))
    if p >= 1.0:
        x_flat = x_samps.reshape(B * T * K_, d)
        c_flat = c.unsqueeze(2).expand(B, T, K_, c.shape[-1]).reshape(B * T * K_, c.shape[-1])
        with autocast_ctx:
            log_q = model.flow.log_prob(x_flat, c_flat).reshape(B, T, K_)
    elif p <= 0.0:
        log_q = None
    else:
        s = max(1, int(K_ * p))
        idx = torch.randperm(K_, device=x_samps.device)[:s]
        x_sub = x_samps.index_select(dim=2, index=idx)  # (B,T,s,d)
        x_flat = x_sub.reshape(B * T * s, d)
        c_flat = c.unsqueeze(2).expand(B, T, s, c.shape[-1]).reshape(B * T * s, c.shape[-1])
        with autocast_ctx:
            log_q_sub = model.flow.log_prob(x_flat, c_flat).reshape(B, T, s)
        log_q = log_q_sub  # 用于熵项；KL 走无 log_q 路径

    loss_obs = obs_negloglik(y, G, x_samps, R)
    loss_prior = prior_kl_mc(x_samps, log_q=None if (p < 1.0) else log_q, alpha=1.0)
    loss_dyn = dyn_smoothness(x_samps, sigma2=1.0)
    loss_entropy = (-log_q.mean()) if (log_q is not None) else torch.tensor(0.0, device=y.device, dtype=y.dtype)

    loss = loss_obs + beta_prior * loss_prior + lambda_dyn * loss_dyn + gamma_entropy * loss_entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), model.cfg.train.grad_clip)
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "obs": float(loss_obs.item()),
        "prior": float(loss_prior.item()),
        "dyn": float(loss_dyn.item()),
        "entropy": float(loss_entropy.item()),
    }


def make_model(cfg: Config, obs_dim: int) -> PosteriorAmortizer:
    # 特征维度：y(m) + vec(G)(m*d) + diag(R)(m)
    f_dim = obs_dim + obs_dim * cfg.model.state_dim + obs_dim
    return PosteriorAmortizer(cfg, in_feat=f_dim)
