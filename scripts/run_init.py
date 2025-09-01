#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import torch

import sys
from pathlib import Path
import matplotlib
# 使用非交互后端，避免无显示环境下绘图失败
matplotlib.use("Agg")

# 确保可以直接运行：将 src/ 加入 Python 路径
_HERE = Path(__file__).resolve()
_SRC = _HERE.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flow_init.config import Config
from flow_init.train import make_model, warmup_epoch, training_epoch
from flow_init.infer import infer_posterior, calibrate_temperature, state_to_obs_moments, obs_loglik
from flow_init.eval import whiten_residual, coverage_curve, pit_values, timewise_coverage


def plot_states(mu_x: torch.Tensor, P_x: torch.Tensor, x_true: torch.Tensor | None, out_path: Path):
    import matplotlib.pyplot as plt
    mu = mu_x[0].cpu().numpy()  # (T,d)
    T, d = mu.shape
    diagP = torch.diagonal(P_x[0], dim1=-2, dim2=-1).cpu().numpy()  # (T,d)
    t = np.arange(T)
    d_plot = min(d, 2)
    fig, axes = plt.subplots(d_plot, 1, figsize=(12, 6), sharex=True)
    if d_plot == 1:
        axes = [axes]
    for i in range(d_plot):
        ax = axes[i]
        mean = mu[:, i]
        std = np.sqrt(np.maximum(diagP[:, i], 0.0))
        ax.plot(t, mean, linestyle="--", color="#8B0000", label="Posterior mean E[x]")
        ax.fill_between(t, mean - 1.96 * std, mean + 1.96 * std, color="#8B0000", alpha=0.15, label="95% CI" if i == 0 else None)
        if x_true is not None:
            xt = x_true[0, :, i].cpu().numpy()
            ax.plot(t, xt, "k-", label="True state x" if i == 0 else None)
        ax.set_ylabel(f"x[{i+1}]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("t")
    fig.suptitle("State space: mean and 95% CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_obs(y: torch.Tensor, mu_y: torch.Tensor, S_y_diag: np.ndarray, out_path: Path):
    import matplotlib.pyplot as plt
    y0 = y[0].cpu().numpy()  # (T,m)
    mu0 = mu_y[0].cpu().numpy()  # (T,m)
    T, m = mu0.shape
    t = np.arange(T)
    d_plot = min(m, 2)
    fig, axes = plt.subplots(d_plot, 1, figsize=(12, 6), sharex=True)
    if d_plot == 1:
        axes = [axes]
    for i in range(d_plot):
        ax = axes[i]
        mean = mu0[:, i]
        std = np.sqrt(np.maximum(S_y_diag[0, :, i], 0.0))
        ax.plot(t, mean, linestyle="--", color="#8B0000", label="E[y]")
        ax.fill_between(t, mean - 1.96 * std, mean + 1.96 * std, color="#8B0000", alpha=0.15, label="95% CI" if i == 0 else None)
        ax.scatter(t, y0[:, i], s=12, c="black", alpha=0.7, label="Observations y" if i == 0 else None)
        ax.set_ylabel(f"y[{i+1}]")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("t")
    fig.suptitle("Observation space: mean and 95% CI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coverage_and_pit(y: torch.Tensor, mu_y: torch.Tensor, S: torch.Tensor, out_curve: Path, out_time: Path):
    import matplotlib.pyplot as plt
    # Coverage curve
    levels = torch.linspace(0.1, 0.99, 30, dtype=torch.float64, device=y.device)
    emp = coverage_curve(y, mu_y, S, levels).cpu().numpy()
    nom = levels.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(nom, nom, "k-", label="Nominal")
    ax.plot(nom, emp, color="#8B0000", label="Empirical")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_curve, dpi=160)
    plt.close(fig)

    # PIT and timewise calibration
    z = whiten_residual(y, mu_y, S)
    pit = 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype, device=z.device))))
    pit_np = pit.cpu().numpy().reshape(-1)
    tw = timewise_coverage(z, q=0.95).cpu().numpy()  # (T,)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # PIT histogram
    axes[0].hist(pit_np, bins=20, color="#8B0000", alpha=0.8, edgecolor="black")
    axes[0].set_title("PIT histogram")
    axes[0].set_xlabel("PIT value")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)
    # Timewise 95% coverage
    T = tw.shape[0]
    axes[1].plot(np.arange(T), tw, color="#8B0000", label="Empirical 95%")
    axes[1].hlines(0.95, 0, T - 1, colors="black", linestyles="--", label="Nominal 95%")
    axes[1].set_title("Timewise coverage (95%)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Coverage")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_time, dpi=160)
    plt.close(fig)


def load_npz(path: str):
    data = np.load(path)
    y = torch.as_tensor(data["y"], dtype=torch.float64)
    G = torch.as_tensor(data["G"], dtype=torch.float64)
    R = torch.as_tensor(data["R"], dtype=torch.float64)
    x = torch.as_tensor(data["x"], dtype=torch.float64) if "x" in data else None
    return y, G, R, x


def synth(B: int, T: int, m: int, d: int, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    G = torch.randn(B, T, m, d, generator=rng, dtype=torch.float64) * 0.8
    x = torch.randn(B, T, d, generator=rng, dtype=torch.float64)
    R = torch.eye(m, dtype=torch.float64) * 0.1
    v = torch.randn(B, T, m, generator=rng, dtype=torch.float64) @ torch.linalg.cholesky(R)
    y = torch.einsum("btmd,btd->btm", G, x) + v
    return y, G, R, x


def main():
    ap = argparse.ArgumentParser(description="运行条件 IAF 初始化")
    ap.add_argument("--npz", type=str, default=None, help="包含 y,G,R 的 npz 路径（不提供则使用合成数据）")
    ap.add_argument("--epochs_warmup", type=int, default=8, help="岭回归预热轮数")
    ap.add_argument("--epochs", type=int, default=40, help="训练轮数（退火）")
    ap.add_argument("--K", type=int, default=512, help="推断采样条数 K")
    ap.add_argument("--device", type=str, default="cpu", help="设备：cpu 或 cuda")
    ap.add_argument("--full_context", type=int, default=1, help="1=双向上下文(可看未来), 0=因果")
    ap.add_argument("--window", type=int, default=-1, help="窗口半径h，-1表示全序列")
    ap.add_argument("--clamp", type=float, default=8.0, help="RealNVP log-scale clamp 上界（对称）")
    ap.add_argument("--min_log_scale", type=float, default=-9.0, help="RealNVP log-scale 下界")
    ap.add_argument("--k_train", type=int, default=16, help="训练期每步采样条数 K_train")
    ap.add_argument("--gamma_entropy", type=float, default=0.02, help="熵系数 γ")
    args = ap.parse_args()

    cfg = Config()
    device = torch.device(args.device)

    if args.npz:
        y, G, R, x_true = load_npz(args.npz)
    else:
        # 最小合成示例（仅用于流程验证）
        B, T, m, d = 2, 100, cfg.model.obs_dim, cfg.model.state_dim
        y, G, R, x_true = synth(B, T, m, d)

    y, G = y.to(device), G.to(device)
    R = R.to(device)

    # 应用 CLI 超参
    cfg.model.flow.clamp = float(args.clamp)
    cfg.model.flow.min_log_scale = float(args.min_log_scale)
    cfg.train.K_train = int(args.k_train)
    cfg.train.gamma_entropy = float(args.gamma_entropy)

    model = make_model(cfg, obs_dim=y.shape[-1]).to(device)
    # 应用 encoder 上下文配置
    model.encoder.cfg.use_full_context = bool(args.full_context)
    model.encoder.cfg.window = None if args.window < 0 else int(args.window)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    print("\n================ 预热阶段 (Ridge) ================")
    for ep in range(args.epochs_warmup):
        wl = warmup_epoch(model, {"y": y, "G": G, "R": R}, cfg.train.ridge_lambda, opt)
        print(f"  预热轮次 {ep+1:02d}/{args.epochs_warmup} | MSE: {wl:.6f}")

    print("\n================ 训练阶段 (退火) ================")
    print(f"  clamp={cfg.model.flow.clamp}, min_log_scale={cfg.model.flow.min_log_scale}, K_train={cfg.train.K_train}, gamma={cfg.train.gamma_entropy}")
    last_stats = None
    for e in range(args.epochs):
        w = (e + 1) / max(1, args.epochs)
        stats = training_epoch(
            model,
            {"y": y, "G": G, "R": R},
            K=cfg.train.K_train,
            beta_prior=w * cfg.train.beta_prior,
            lambda_dyn=w * cfg.train.lambda_dyn,
            gamma_entropy=cfg.train.gamma_entropy,
            optimizer=opt,
        )
        last_stats = stats
        print(
            f"  训练轮次 {e+1:02d}/{args.epochs} | 总损失 {stats['loss']:.6f} | "
            f"obs {stats['obs']:.6f} | prior {stats['prior']:.6f} | dyn {stats['dyn']:.6f} | H {stats['entropy']:.6f}"
        )

    out = infer_posterior(cfg, model.encoder, model.flow, y, G, R, K=args.K, eps_reg=cfg.infer.eps_reg)
    mu, P = out["mu"], out["P"]
    tau = calibrate_temperature(mu, P, y, G, R)
    ll = obs_loglik(mu, P, y, G, R, tau=tau)
    mu_y, S_y = state_to_obs_moments(mu, tau * P, G, R)

    print("\n================ 推断与指标 ================")
    print(f"  数据形状: y={tuple(y.shape)}, G={tuple(G.shape)}, R={tuple(R.shape)}")
    print(f"  模型维度: d(state)={cfg.model.state_dim}, m(obs)={y.shape[-1]}, K={args.K}")
    if last_stats is not None:
        print(
            f"  最后一轮训练: loss={last_stats['loss']:.6f}, obs={last_stats['obs']:.6f}, prior={last_stats['prior']:.6f}, dyn={last_stats['dyn']:.6f}, H={last_stats['entropy']:.6f}"
        )
    print(f"  温度校准 tau: {tau:.4f}")
    print(f"  平均观测对数似然（按 B,T 平均）: {ll.item():.6f}")

    # Save results npz in current directory
    out_path = Path.cwd() / "init_outputs.npz"
    np.savez(
        out_path,
        mu=mu.cpu().numpy(),
        P=P.cpu().numpy(),
        tau=np.asarray(tau),
        mu_y=mu_y.cpu().numpy(),
        # S_y may be large; save diagonal only by default
        S_y_diag=np.diagonal(S_y.cpu().numpy(), axis1=-2, axis2=-1),
    )
    S_diag = np.diagonal(S_y.cpu().numpy(), axis1=-2, axis2=-1)
    fig_state = Path.cwd() / "state_plot.png"
    fig_obs = Path.cwd() / "obs_plot.png"
    try:
        # 使用温度校准后的协方差绘制状态置信区间，更贴近观测域校准结果
        plot_states(mu, tau * P, x_true, fig_state)
        plot_obs(y, mu_y, S_diag, fig_obs)
        # Calibration plots
        fig_cov = Path.cwd() / "coverage_curve.png"
        fig_pit = Path.cwd() / "pit_timewise.png"
        plot_coverage_and_pit(y, mu_y, S_y, fig_cov, fig_pit)
        print(
            f"\n================ 保存结果 ================\n  输出文件: {out_path}\n  图像: {fig_state}, {fig_obs}, {fig_cov}, {fig_pit}\n  包含键: mu, P, tau, mu_y, S_y_diag\n"
        )
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(f"绘图失败（将跳过图像）：\n{tb}\n数值已保存到 {out_path}")


if __name__ == "__main__":
    main()
