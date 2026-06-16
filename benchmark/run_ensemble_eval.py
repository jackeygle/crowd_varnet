"""
Deep Ensemble 综合评测 + 300 帧可视化。

输出:
  - summary_table.json       (所有方法指标)
  - per_step_rmse.png        (per-step RMSE 曲线)
  - uncertainty_plot.png     (aleatoric / epistemic / total 随步数变化)
  - visualizations/vis_XXXX.png  (300 帧可视化，含不确定度热力图)

用法:
    python -m benchmark.run_ensemble_eval --device cuda --num-steps 300 \
        --save-dir benchmark/results/ensemble
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import (
    PROJECT_ROOT, NUM_AGENTS, SENSING_RANGE, RANDOM_SEED, RHO_MASK_THR,
    GRID_SIZE, STATE_SHAPE, WARMUP_STEPS,
    OUR_TEACHER_CKPT, OUR_TEACHER_ARCH,
    THEIR_TEACHER_CKPT, CVN_BEST_CKPT, PO_ROOT,
)
from .data_loader import get_test_sequence
from .obs_generator import create_observation_sequence
from .metrics import compute_all_metrics


# ============================================================
# Ensemble method wrapper
# ============================================================
class CrowdVarNetEnsembleMethod:
    """Deep Ensemble of 5 CrowdVarNet students with uncertainty decomposition."""
    name = "CrowdVarNet Ensemble (ours)"

    def __init__(self, ckpt_paths: List[str], device: str = "cuda"):
        cvn_root = str(PROJECT_ROOT)
        if cvn_root not in sys.path:
            sys.path.insert(0, cvn_root)

        from crowd_varnet.cli._common import build_model_from_ckpt

        self.device = torch.device(device)
        self.members = []
        for p in ckpt_paths:
            model, meta = build_model_from_ckpt(Path(p), device=self.device)
            model.eval()
            self.members.append(model)
        self.T_hist = self.members[0].T_hist
        self.history_buf = None
        print(f"  [Ensemble] Loaded {len(self.members)} members", flush=True)

    def initialize(self, first_frame: np.ndarray) -> None:
        frame_t = torch.from_numpy(first_frame).float().unsqueeze(0)
        self.history_buf = frame_t.repeat(1, self.T_hist, 1, 1, 1).to(self.device)

    def step(self, obs_dict: dict, true_frame: np.ndarray = None):
        """Returns (x_hat_mean, aleatoric, epistemic, total_var) all as numpy [C, H, W]."""
        obs_mask = torch.from_numpy(obs_dict["obs_mask"]).float().unsqueeze(0).to(self.device)
        obs = torch.from_numpy(obs_dict["obs_clean"]).float().unsqueeze(0).to(self.device)

        means = []
        log_vars = []

        for m in self.members:
            saved = [(p, p.requires_grad) for p in m.parameters() if p.requires_grad]
            for p, _ in saved:
                p.requires_grad_(False)
            try:
                with torch.set_grad_enabled(True):
                    x_hat, log_var = m.forward_with_var(self.history_buf, obs, obs_mask)
            finally:
                for p, was in saved:
                    p.requires_grad_(was)
            means.append(x_hat.detach())
            if log_var is not None:
                log_vars.append(log_var.detach())

        means_t = torch.stack(means, dim=0)  # [N, 1, 4, H, W]
        ens_mean = means_t.mean(dim=0)       # [1, 4, H, W]

        # Epistemic: variance of means (first 3 channels)
        epistemic = means_t[:, :, :3].var(dim=0, unbiased=False)  # [1, 3, H, W]

        # Aleatoric: mean of exp(log_var)
        if log_vars:
            log_var_t = torch.stack(log_vars, dim=0)  # [N, 1, 3, H, W]
            aleatoric = torch.exp(log_var_t).mean(dim=0)  # [1, 3, H, W]
        else:
            aleatoric = torch.zeros_like(epistemic)

        total_var = aleatoric + epistemic

        # Update history with ensemble mean
        self.history_buf = torch.cat(
            [self.history_buf[:, 1:], ens_mean.unsqueeze(1)], dim=1
        )

        return (
            ens_mean[0].cpu().numpy().astype(np.float64),
            aleatoric[0].cpu().numpy().astype(np.float64),
            epistemic[0].cpu().numpy().astype(np.float64),
            total_var[0].cpu().numpy().astype(np.float64),
        )


# ============================================================
# Visualization
# ============================================================
def plot_frame(
    step_idx: int,
    x_gt: np.ndarray,
    obs: np.ndarray,
    obs_mask: np.ndarray,
    ens_mean: np.ndarray,
    aleatoric: np.ndarray,
    epistemic: np.ndarray,
    enkf_pred: Optional[np.ndarray],
    save_path: Path,
):
    """One frame: GT / Obs / Ensemble Mean / EnKF / Aleatoric / Epistemic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cols = 6 if enkf_pred is not None else 5
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    def _plot_density(ax, data, title, cmap="Blues", vmin=0, vmax=1.0):
        im = ax.imshow(data, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    # GT density
    _plot_density(axes[0], x_gt[0], f"GT (step {step_idx})")

    # Partial obs
    obs_vis = obs[0].copy()
    obs_vis[obs_mask[0] < 0.5] = np.nan
    _plot_density(axes[1], obs_vis, "Partial Obs")

    # Ensemble mean density
    _plot_density(axes[2], ens_mean[0], "Ensemble Mean")

    col = 3
    if enkf_pred is not None:
        _plot_density(axes[col], enkf_pred[0], "EnKF")
        col += 1

    # Aleatoric (sqrt for σ)
    aleatoric_rho = np.sqrt(aleatoric[0].clip(0)) if aleatoric.shape[0] > 0 else np.zeros_like(x_gt[0])
    _plot_density(axes[col], aleatoric_rho, "Aleatoric σ(ρ)", cmap="Reds", vmin=0, vmax=0.5)
    col += 1

    # Epistemic (sqrt for σ)
    epistemic_rho = np.sqrt(epistemic[0].clip(0)) if epistemic.shape[0] > 0 else np.zeros_like(x_gt[0])
    _plot_density(axes[col], epistemic_rho, "Epistemic σ(ρ)", cmap="Oranges", vmin=0, vmax=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Deep Ensemble evaluation + 300 frame visualization")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--save-dir", type=str, default="benchmark/results/ensemble")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--run-enkf", action="store_true", help="Also run EnKF for comparison")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = save_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Discover ensemble checkpoints
    ensemble_ckpts = sorted([
        str(p) for p in PROJECT_ROOT.glob("runs/cvn_ensemble_seed*_*/best.pt")
    ])
    print(f"[ensemble-eval] Found {len(ensemble_ckpts)} ensemble members:", flush=True)
    for p in ensemble_ckpts:
        print(f"  {p}", flush=True)
    if len(ensemble_ckpts) < 2:
        print("ERROR: Need at least 2 ensemble members. Exiting.", flush=True)
        sys.exit(1)

    # Load data
    print(f"\n[ensemble-eval] Loading data (split={args.split}, steps={args.num_steps})...", flush=True)
    frames = get_test_sequence(split=args.split, max_steps=args.num_steps)
    print(f"[ensemble-eval] Loaded {len(frames)} frames", flush=True)

    print(f"[ensemble-eval] Generating observations (seed={RANDOM_SEED})...", flush=True)
    observations = create_observation_sequence(frames, seed=RANDOM_SEED)

    # Build ensemble
    print(f"\n[ensemble-eval] Building ensemble...", flush=True)
    ensemble = CrowdVarNetEnsembleMethod(ensemble_ckpts, device=args.device)
    ensemble.initialize(frames[0])

    # Optionally build EnKF
    enkf = None
    if args.run_enkf:
        print(f"[ensemble-eval] Building EnKF...", flush=True)
        from .methods import EnKFMethod
        enkf = EnKFMethod(device="cpu")
        enkf.initialize(frames[0])

    # Run evaluation
    print(f"\n[ensemble-eval] Running {args.num_steps} steps...", flush=True)
    ens_metrics = []
    enkf_metrics = []
    aleatoric_history = []
    epistemic_history = []

    for step_idx in range(len(frames)):
        obs_dict = observations[step_idx]

        # Ensemble
        t0 = time.time()
        ens_mean, aleatoric, epistemic, total_var = ensemble.step(obs_dict, frames[step_idx])
        ens_time = time.time() - t0

        m = compute_all_metrics(ens_mean, frames[step_idx], RHO_MASK_THR)
        m["time_s"] = ens_time
        m["step"] = step_idx
        ens_metrics.append(m)

        # Track uncertainty stats (support pixels only)
        support = frames[step_idx][0] > RHO_MASK_THR
        if support.sum() > 0:
            aleatoric_history.append(float(np.sqrt(aleatoric[0][support].mean())))
            epistemic_history.append(float(np.sqrt(epistemic[0][support].mean())))
        else:
            aleatoric_history.append(0.0)
            epistemic_history.append(0.0)

        # EnKF
        enkf_pred = None
        if enkf is not None:
            t0 = time.time()
            enkf_pred = enkf.step(obs_dict, frames[step_idx])
            enkf_time = time.time() - t0
            em = compute_all_metrics(enkf_pred, frames[step_idx], RHO_MASK_THR)
            em["time_s"] = enkf_time
            em["step"] = step_idx
            enkf_metrics.append(em)

        # Visualization (every frame = 300 images)
        plot_frame(
            step_idx=step_idx,
            x_gt=frames[step_idx],
            obs=obs_dict["obs_clean"],
            obs_mask=obs_dict["obs_mask"],
            ens_mean=ens_mean,
            aleatoric=aleatoric,
            epistemic=epistemic,
            enkf_pred=enkf_pred,
            save_path=viz_dir / f"vis_{step_idx:04d}.png",
        )

        if (step_idx + 1) % 25 == 0:
            print(
                f"  step {step_idx+1}/{len(frames)}  "
                f"RMSE={m['full_rmse']:.4f}  "
                f"σ_ale={aleatoric_history[-1]:.4f}  "
                f"σ_epi={epistemic_history[-1]:.4f}  "
                f"time={ens_time:.3f}s",
                flush=True,
            )

    # === Summary ===
    skip = 10
    steady_ens = ens_metrics[skip:]
    summary = {
        "ensemble": {
            "full_rmse": float(np.mean([m["full_rmse"] for m in steady_ens])),
            "rho": float(np.mean([m["rmse_support_rho"] for m in steady_ens])),
            "vx": float(np.mean([m["rmse_support_vx"] for m in steady_ens])),
            "vy": float(np.mean([m["rmse_support_vy"] for m in steady_ens])),
            "time": float(np.mean([m["time_s"] for m in ens_metrics])),
            "aleatoric_mean": float(np.mean(aleatoric_history[skip:])),
            "epistemic_mean": float(np.mean(epistemic_history[skip:])),
            "n_members": len(ensemble_ckpts),
        }
    }
    if enkf_metrics:
        steady_enkf = enkf_metrics[skip:]
        summary["enkf"] = {
            "full_rmse": float(np.mean([m["full_rmse"] for m in steady_enkf])),
            "rho": float(np.mean([m["rmse_support_rho"] for m in steady_enkf])),
            "vx": float(np.mean([m["rmse_support_vx"] for m in steady_enkf])),
            "vy": float(np.mean([m["rmse_support_vy"] for m in steady_enkf])),
            "time": float(np.mean([m["time_s"] for m in enkf_metrics])),
        }

    print(f"\n{'='*70}")
    print("ENSEMBLE EVALUATION SUMMARY (support pixels, skip first 10 steps)")
    print(f"{'='*70}")
    for k, v in summary.items():
        print(f"\n  {k}:")
        for mk, mv in v.items():
            print(f"    {mk}: {mv}")

    (save_dir / "summary_table.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {save_dir / 'summary_table.json'}")

    # === Per-step RMSE plot ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    channels = [("full_rmse", "Full RMSE"), ("rmse_support_rho", "ρ RMSE"),
                ("rmse_support_vx", "vx RMSE"), ("rmse_support_vy", "vy RMSE")]
    for ax, (key, title) in zip(axes.flat, channels):
        steps = [m["step"] for m in ens_metrics][skip:]
        vals = [m[key] for m in ens_metrics][skip:]
        ax.plot(steps, vals, label="Ensemble", color="blue", linewidth=1.5)
        if enkf_metrics:
            vals_e = [m[key] for m in enkf_metrics][skip:]
            ax.plot(steps, vals_e, label="EnKF", color="red", linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "per_step_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir / 'per_step_rmse.png'}")

    # === Uncertainty plot ===
    fig, ax = plt.subplots(figsize=(10, 5))
    steps_arr = list(range(len(aleatoric_history)))
    ax.plot(steps_arr[skip:], aleatoric_history[skip:], label="Aleatoric σ(ρ)", color="red", linewidth=1.5)
    ax.plot(steps_arr[skip:], epistemic_history[skip:], label="Epistemic σ(ρ)", color="orange", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("σ (std dev)")
    ax.set_title("Uncertainty Decomposition over Time (support pixels)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_dir / "uncertainty_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_dir / 'uncertainty_plot.png'}")

    # === Per-step metrics JSON ===
    (save_dir / "per_step_metrics.json").write_text(json.dumps({
        "ensemble": ens_metrics,
        "enkf": enkf_metrics if enkf_metrics else None,
        "aleatoric_history": aleatoric_history,
        "epistemic_history": epistemic_history,
    }, indent=2))
    print(f"  Saved: {save_dir / 'per_step_metrics.json'}")

    print(f"\n[ensemble-eval] Done! {len(frames)} visualizations in {viz_dir}", flush=True)


if __name__ == "__main__":
    main()
