"""教师 inference 可视化：跑教师在 val 数据上做单步预测，输出 PNG 对比图。

每张 PNG 4 列：(GT density, pred density, GT velocity quiver, pred velocity quiver)

用法::
    python -m scripts.teacher_infer_viz --ckpt /path/best.hkl --arch pedpred3_gru \\
        --out-dir /path/out --n-frames 100
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from crowd_varnet.models.prior import load_frozen_pedpred
from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.deps.grid_data import GridData


def _plot_density(ax, density, title, vmin=0.0, vmax=1.0):
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_under("white")
    im = ax.imshow(density, cmap=cmap, aspect="auto", origin="upper",
                   vmin=max(vmin, 1e-3), vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _plot_state(ax, density, vx, vy, title, vmin=0.0, vmax=1.0):
    """Density + velocity quiver overlay."""
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_under("white")
    H, W = density.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    speed = np.sqrt(vx**2 + vy**2)
    u = np.where(speed < 1e-6, np.nan, vx / (speed + 1e-9))
    v = np.where(speed < 1e-6, np.nan, vy / (speed + 1e-9))
    nan_mask = np.isnan(density)
    u[nan_mask] = np.nan
    v[nan_mask] = np.nan
    d = density.copy()
    d[nan_mask] = 0.0
    im = ax.imshow(d, cmap=cmap, aspect="auto", origin="upper",
                   vmin=max(vmin, 1e-3), vmax=vmax)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _save_frame_plot(gt_state, pred_state, idx, out_dir, vmax=1.0):
    """4 列图：GT state, Pred state, |Pred-GT| density, |Pred-GT| velocity"""
    gt_d, gt_vx, gt_vy = gt_state[0], gt_state[1], gt_state[2]
    pr_d, pr_vx, pr_vy = pred_state[0], pred_state[1], pred_state[2]

    den_err = np.abs(pr_d - gt_d)
    vel_err = np.sqrt((pr_vx - gt_vx)**2 + (pr_vy - gt_vy)**2)

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15)

    ax0 = fig.add_subplot(gs[0])
    im0 = _plot_state(ax0, gt_d, gt_vx, gt_vy, f"Step {idx} - GT state", vmax=vmax)
    ax1 = fig.add_subplot(gs[1])
    im1 = _plot_state(ax1, pr_d, pr_vx, pr_vy, "Pred state", vmax=vmax)
    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(den_err, cmap="Reds", aspect="auto", origin="upper", vmin=0)
    ax2.set_title(f"|Pred-GT| Density\nmean={float(den_err.mean()):.4f}  max={float(den_err.max()):.4f}")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(vel_err, cmap="Reds", aspect="auto", origin="upper", vmin=0)
    ax3.set_title(f"|Pred-GT| Velocity\nmean={float(vel_err.mean()):.4f}  max={float(vel_err.max()):.4f}")
    ax3.set_xticks([]); ax3.set_yticks([])

    for ax, im in [(ax0, im0), (ax1, im1), (ax2, im2), (ax3, im3)]:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"step_{idx:03d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="pedpred3_gru")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-frames", type=int, default=100)
    ap.add_argument("--rho-vmax", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    print(f"[teacher_infer_viz] ckpt={args.ckpt} arch={args.arch} out={out_dir}")

    teacher = load_frozen_pedpred(args.ckpt, device, arch=args.arch)
    teacher.eval()

    # 用 valid 集合，nin=5, nout=1
    val_loader = get_atc_data("valid", batch=1, nin=5, nout=1,
                              num_workers=0, drop_last=False, pin_memory=False)

    saved = 0
    with torch.no_grad():
        for inp, tgt in val_loader:
            if saved >= args.n_frames:
                break
            inp = inp.to(device)
            tgt = tgt.to(device)
            inp_t = GridData(inp).as_tensor("density", "vel_mean", "vel_var").to(device)
            tgt_t = GridData(tgt).as_tensor("density", "vel_mean", "vel_var").to(device)

            out = teacher(inp_t, horizon=1)
            pred_t = GridData(out).as_tensor("density", "vel_mean", "vel_var").to(device)

            # squeeze T dim (horizon=1)
            gt = tgt_t[0, 0].detach().cpu().numpy()    # [4, H, W]
            pred = pred_t[0, 0].detach().cpu().numpy() # [4, H, W]

            _save_frame_plot(gt, pred, saved, out_dir, vmax=args.rho_vmax)
            saved += 1
            if saved % 10 == 0:
                print(f"  saved {saved}/{args.n_frames}", flush=True)

    print(f"[teacher_infer_viz] done. {saved} PNGs saved to {out_dir}")


if __name__ == "__main__":
    main()
