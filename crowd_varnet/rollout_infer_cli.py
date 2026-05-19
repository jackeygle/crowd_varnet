"""CrowdVarNet rollout 推理 + 可视化。

与训练时数据流完全一致：
  - history 每步用自己的预测 x_hat（不用 GT）
  - obs 来自 RolloutEpisodeDataset（agent 连续移动）
  - 输出：每个 episode 的逐帧 PNG + 每步 MSE 曲线 + summary.json

用法::

    python -m crowd_varnet.rollout_infer_cli \\
        --ckpt /path/to/best.pt \\
        --save-dir /path/to/output \\
        --n-episodes 5 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from crowd_varnet.cli import build_model_from_ckpt
from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.datasets import RolloutEpisodeDataset, unwrap_concat_base_dataset
from crowd_varnet.models.cost import density_support_mask


CH_NAMES = ("rho", "vx", "vy", "speed")


def _plot_state(state: np.ndarray, ax, *, vmin: float = 0.0, vmax: float = 1.0):
    density = state[0].copy()
    vx = state[1]
    vy = state[2]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    H, W = density.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    u = np.where(speed < 1e-6, np.nan, vx / (speed + 1e-9))
    v = np.where(speed < 1e-6, np.nan, vy / (speed + 1e-9))
    nan_mask = np.isnan(density)
    u[nan_mask] = np.nan
    v[nan_mask] = np.nan
    density[nan_mask] = 0.0
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_under("white")
    im = ax.imshow(density, cmap=cmap, aspect="auto", origin="upper", vmin=max(vmin, 1e-3), vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def _save_step_plot(
    partial_obs: np.ndarray,
    true_state: np.ndarray,
    estimated: np.ndarray,
    step_idx: int,
    out_dir: Path,
    density_vmax: float = 1.0,
):
    density_err = np.abs(estimated[0] - true_state[0])
    emean = float(np.nanmean(density_err))
    emax = float(np.nanmax(density_err))

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15)

    ax0 = fig.add_subplot(gs[0])
    im0 = _plot_state(partial_obs, ax0, vmax=density_vmax)
    ax0.set_title(f"Step {step_idx} — Partial Obs")

    ax1 = fig.add_subplot(gs[1])
    im1 = _plot_state(true_state, ax1, vmax=density_vmax)
    ax1.set_title("True State")

    ax2 = fig.add_subplot(gs[2])
    im2 = _plot_state(estimated, ax2, vmax=density_vmax)
    ax2.set_title("CrowdVarNet (rollout)")

    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(density_err, cmap="Reds", origin="upper", aspect="auto")
    ax3.set_title(f"|Density Error|\nmean={emean:.4f}  max={emax:.4f}")
    ax3.set_xticks([])
    ax3.set_yticks([])

    for ax, im in [(ax0, im0), (ax1, im1), (ax2, im2), (ax3, im3)]:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"step_{step_idx:03d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_loss_curve(per_ch_mse: np.ndarray, save_path: Path, ep_idx: int, warmup: int):
    L = per_ch_mse.shape[0]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for ch in range(4):
        ax.plot(np.arange(warmup, L), per_ch_mse[warmup:, ch], label=CH_NAMES[ch])
    ax.axvline(warmup, color="gray", linestyle="--", linewidth=0.8, label="warmup end")
    ax.set_xlabel("rollout step t")
    ax.set_ylabel("masked MSE per channel")
    ax.set_title(f"episode {ep_idx}: per-step MSE (CrowdVarNet rollout)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _rollout_one_episode(
    model, batch, device: torch.device, warmup: int, rho_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rollout 一个 episode，返回 (x_gt, obs, mask, x_hat_seq, per_ch_mse)。"""
    x_gt_seq, obs_seq, mask_seq = batch
    x_gt_seq = x_gt_seq.to(device)   # [1, L, 4, H, W]
    obs_seq  = obs_seq.to(device)
    mask_seq = mask_seq.to(device)
    B, L, C, H, W = x_gt_seq.shape
    assert B == 1

    history_buf = x_gt_seq[:, :warmup].clone()
    x_hat_seq = x_gt_seq.clone()
    per_ch_mse = np.zeros((L, 4), dtype=np.float32)

    # solver 需要 autograd，不能用 torch.no_grad()
    saved = [(p, p.requires_grad) for p in model.parameters() if p.requires_grad]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        for t in range(warmup, L):
            x_hat = model.forward(history_buf, obs_seq[:, t], mask_seq[:, t]).detach()
            x_hat_seq[:, t] = x_hat
            err = (x_hat - x_gt_seq[:, t]).pow(2)
            dm = density_support_mask(x_gt_seq[:, t], rho_thr)
            per_ch_mse[t] = (
                (err * dm).sum(dim=(0, 2, 3)) / dm.sum().clamp_min(1e-6)
            ).cpu().numpy()
            history_buf = torch.cat([history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1)
    finally:
        for p, was in saved:
            p.requires_grad_(was)

    return (
        x_gt_seq[0].cpu().numpy(),
        obs_seq[0].cpu().numpy(),
        mask_seq[0].cpu().numpy(),
        x_hat_seq[0].cpu().numpy(),
        per_ch_mse,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="CrowdVarNet rollout inference + visualization")
    ap.add_argument("--ckpt", required=True, help="best.pt / last.pt")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--frames-per-plot", type=int, default=10)
    ap.add_argument("--episode-len", type=int, default=150)
    ap.add_argument("--density-vmax", type=float, default=1.0)
    ap.add_argument("--val-workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    model, meta = build_model_from_ckpt(ckpt_path, device=device)
    model.eval()
    warmup    = int(meta.get("warmup", model.T_hist))
    s_range   = float(meta.get("sensing_range", 5.0))
    n_agents  = int(meta.get("num_agents", 3))
    rho_thr   = float(meta.get("rho_mask_thr", 0.05))
    print(f"[rollout-infer] ckpt={ckpt_path.name}  warmup={warmup}  "
          f"sensing_range={s_range}  num_agents={n_agents}", flush=True)

    _, base_val = get_atc_data(
        "train", "valid", batch=1, nin=warmup, nout=1,
        num_workers=0, validation_num_workers=0,
    )
    bases = list(unwrap_concat_base_dataset(base_val.dataset))
    wrapped = [
        RolloutEpisodeDataset(
            b, episode_len=args.episode_len,
            sensing_range=s_range, num_agents=n_agents,
            seed=113 + i * 7919,
        ) for i, b in enumerate(bases)
    ]
    ds = wrapped[0] if len(wrapped) == 1 else ConcatDataset(wrapped)
    val_loader = DataLoader(dataset=ds, batch_size=1, shuffle=False,
                            num_workers=args.val_workers)
    print(f"[rollout-infer] val_episodes={len(val_loader.dataset)}  "
          f"running {args.n_episodes}", flush=True)

    summaries = []
    for ep_idx, batch in enumerate(val_loader):
        if ep_idx >= args.n_episodes:
            break
        print(f"[rollout-infer] episode {ep_idx} ...", flush=True)
        x_gt, obs, mask, x_hat, per_ch_mse = _rollout_one_episode(
            model, batch, device, warmup, rho_thr,
        )
        L = x_gt.shape[0]
        idxs = np.linspace(warmup, L - 1, args.frames_per_plot).astype(int).tolist()
        ep_dir = save_dir / f"ep{ep_idx:02d}"

        for t in idxs:
            obs_frame = obs[t].copy()
            obs_frame[:, mask[t, 0] < 0.5] = np.nan
            _save_step_plot(obs_frame, x_gt[t], x_hat[t], t, ep_dir, args.density_vmax)

        _save_loss_curve(per_ch_mse, save_dir / f"ep{ep_idx:02d}_loss_curve.png",
                         ep_idx, warmup)

        valid = per_ch_mse[warmup:]
        ep_summary = {
            "episode": ep_idx,
            "mean_mse": {CH_NAMES[c]: float(valid[:, c].mean()) for c in range(4)},
            "final_mse": {CH_NAMES[c]: float(valid[-1, c]) for c in range(4)},
            "frames_plotted": idxs,
        }
        summaries.append(ep_summary)
        print(f"  mean MSE: rho={ep_summary['mean_mse']['rho']:.4f} "
              f"vx={ep_summary['mean_mse']['vx']:.4f} "
              f"vy={ep_summary['mean_mse']['vy']:.4f} "
              f"sp={ep_summary['mean_mse']['speed']:.4f}", flush=True)

    (save_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(f"[rollout-infer] done -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
