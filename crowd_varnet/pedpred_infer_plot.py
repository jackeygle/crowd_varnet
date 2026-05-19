"""PedPred-only open-loop rollout inference + visualization.

Loads the trained PedPred checkpoint (no solver, no observations assimilated),
runs open-loop rollout on N val episodes, saves Partial_observation-style
plots (density Blues + velocity quiver) for direct comparison with v3 results.

At each step t ≥ warmup:
    x_hat = PedPred(history)        # pure forecast, ignores obs
    history ← [history[1:], x_hat]
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

from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.datasets import RolloutEpisodeDataset, unwrap_concat_base_dataset
from crowd_varnet.models import load_frozen_pedpred
from crowd_varnet.models.prior import FrozenPedPredPrior


CH_NAMES = ("rho", "vx", "vy", "speed")


def _density_mask(x_gt: torch.Tensor, rho_thr: float) -> torch.Tensor:
    return (x_gt[:, 0:1] > rho_thr).to(dtype=x_gt.dtype)


def plot_generated_matrix_on_ax(state: np.ndarray, ax, *, vmin: float = 0.0, vmax: float = 1.0):
    density = state[0]
    vx = state[1]
    vy = state[2]
    speed = np.sqrt(vx ** 2 + vy ** 2)
    heading = np.arctan2(vy, vx)
    H, W = density.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    u = np.cos(heading)
    v = np.sin(heading)
    mask = np.isnan(density) | (speed < 1e-6)
    u = np.where(mask, np.nan, u)
    v = np.where(mask, np.nan, v)
    im = ax.imshow(density, cmap="Blues", aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_step_plot(
    partial_obs: np.ndarray, true_state: np.ndarray,
    estimated_mean: np.ndarray, density_error: np.ndarray,
    step_idx: int, run_dir: Path, density_vmax: float = 1.0,
):
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15)
    ax0 = fig.add_subplot(gs[0])
    im0 = plot_generated_matrix_on_ax(partial_obs, ax0, vmax=density_vmax)
    ax0.set_title(f"Step {step_idx} — Partial Obs")
    ax1 = fig.add_subplot(gs[1])
    im1 = plot_generated_matrix_on_ax(true_state, ax1, vmax=density_vmax)
    ax1.set_title("True State")
    ax2 = fig.add_subplot(gs[2])
    im2 = plot_generated_matrix_on_ax(estimated_mean, ax2, vmax=density_vmax)
    ax2.set_title("PedPred (no obs)")
    ax3 = fig.add_subplot(gs[3])
    ax3.set_title("|Density Error|")
    err = density_error.copy()
    err[np.isnan(err)] = 0
    im3 = ax3.imshow(err, cmap="Reds", origin="upper", aspect="auto")
    ax3.set_xticks([]); ax3.set_yticks([])
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    fname = run_dir / f"step_{step_idx:03d}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_rollout_loss(per_channel_loss_by_t: np.ndarray, save_path: Path, ep_idx: int):
    L = per_channel_loss_by_t.shape[0]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for ch in range(4):
        ax.plot(np.arange(L), per_channel_loss_by_t[:, ch], label=CH_NAMES[ch])
    ax.set_xlabel("rollout step t")
    ax.set_ylabel("masked MSE per channel")
    ax.set_title(f"episode {ep_idx}: per-step MSE  (PedPred open-loop)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def _rollout_one_episode(
    prior: FrozenPedPredPrior, batch, device, warmup: int, rho_thr: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_gt_seq, obs_seq, mask_seq = batch
    x_gt_seq = x_gt_seq.to(device)
    obs_seq = obs_seq.to(device)
    mask_seq = mask_seq.to(device)
    B, L, C, H, W = x_gt_seq.shape
    assert B == 1
    history_buf = x_gt_seq[:, :warmup].clone()
    x_hat_seq = x_gt_seq.clone()
    per_ch_mse = np.zeros((L, 4), dtype=np.float32)
    for t in range(warmup, L):
        x_hat = prior(history_buf)                # [B, 4, H, W]
        x_hat_seq[:, t] = x_hat
        err = (x_hat - x_gt_seq[:, t]).pow(2)
        dm = _density_mask(x_gt_seq[:, t], rho_thr)
        per_ch_mse[t] = (
            (err * dm).sum(dim=(0, 2, 3)) / dm.sum().clamp_min(1e-6)
        ).cpu().numpy()
        history_buf = torch.cat([history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1)
    return (
        x_gt_seq[0].cpu().numpy(),
        obs_seq[0].cpu().numpy(),
        mask_seq[0].cpu().numpy(),
        x_hat_seq[0].cpu().numpy(),
        per_ch_mse,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="PedPred .hkl / .pth")
    ap.add_argument("--arch", default="pedpred3")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-episodes", type=int, default=3)
    ap.add_argument("--frames-per-plot", type=int, default=8)
    ap.add_argument("--episode-len", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--sensing-range", type=float, default=5.0)
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--rho-mask-thr", type=float, default=0.05)
    ap.add_argument("--density-vmax", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    ped = load_frozen_pedpred(args.checkpoint, device, arch=args.arch)
    prior = FrozenPedPredPrior(ped, freeze=True).to(device).eval()
    print(f"[pedpred-infer] loaded {args.checkpoint}", flush=True)

    _, base_val = get_atc_data(
        "train", "valid", batch=1, nin=args.warmup, nout=1,
        num_workers=0, validation_num_workers=0,
    )
    bases = list(unwrap_concat_base_dataset(base_val.dataset))
    wrapped = [
        RolloutEpisodeDataset(
            b, episode_len=args.episode_len,
            sensing_range=args.sensing_range, num_agents=args.num_agents,
            seed=113 + i * 7919,
        ) for i, b in enumerate(bases)
    ]
    ds = wrapped[0] if len(wrapped) == 1 else ConcatDataset(wrapped)
    val_loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"[pedpred-infer] val_episodes={len(val_loader.dataset)} rolling {args.n_episodes}", flush=True)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for ep_idx, batch in enumerate(val_loader):
        if ep_idx >= args.n_episodes:
            break
        print(f"[pedpred-infer] episode {ep_idx} ...", flush=True)
        x_gt, obs, mask, x_hat, per_ch_mse = _rollout_one_episode(
            prior, batch, device, args.warmup, args.rho_mask_thr,
        )
        L = x_gt.shape[0]
        idxs = np.linspace(args.warmup, L - 1, args.frames_per_plot).astype(int).tolist()
        ep_dir = save_dir / f"ep{ep_idx:02d}"
        ep_dir.mkdir(exist_ok=True)
        for t in idxs:
            obs_frame = obs[t].copy()
            m = mask[t, 0]
            obs_frame[:, m < 0.5] = np.nan
            density_err = np.abs(x_hat[t, 0] - x_gt[t, 0])
            save_step_plot(
                partial_obs=obs_frame, true_state=x_gt[t],
                estimated_mean=x_hat[t], density_error=density_err,
                step_idx=t, run_dir=ep_dir, density_vmax=args.density_vmax,
            )
        _plot_rollout_loss(per_ch_mse, save_dir / f"ep{ep_idx:02d}_rollout_loss.png", ep_idx)

        valid = per_ch_mse[args.warmup:]
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

    (save_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"[pedpred-infer] done -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
