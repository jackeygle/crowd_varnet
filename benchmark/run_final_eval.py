"""
最终综合评测 + 生成论文用图表。

输出:
  - summary_table.json  (所有方法的指标)
  - per_step_rmse.png   (per-step RMSE 曲线)
  - vis_step_XXX.png    (可视化对比帧)

用法:
    python -m benchmark.run_final_eval --device cuda --num-steps 100 --save-dir benchmark/results/final
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import NUM_AGENTS, SENSING_RANGE, RANDOM_SEED, RHO_MASK_THR
from .data_loader import get_test_sequence
from .obs_generator import create_observation_sequence
from .metrics import compute_all_metrics


def _run_method(method, frames, observations):
    """Run a method, return per-step metrics."""
    method.initialize(frames[0])
    per_step = []
    for step_idx in range(len(frames)):
        obs_dict = observations[step_idx]
        t0 = time.time()
        x_hat = method.step(obs_dict, frames[step_idx])
        elapsed = time.time() - t0
        m = compute_all_metrics(x_hat, frames[step_idx], RHO_MASK_THR)
        m["time_s"] = elapsed
        m["step"] = step_idx
        per_step.append(m)
    return per_step


def plot_per_step_rmse(all_results: Dict, save_path: Path, skip: int = 5):
    """Plot per-step RMSE curves for all methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    channels = [("full_rmse", "Full RMSE (support pixels)"),
                ("rmse_support_rho", "Density ρ RMSE"),
                ("rmse_support_vx", "Velocity vx RMSE"),
                ("rmse_support_vy", "Velocity vy RMSE")]

    colors = {"enkf": "red", "cvn": "blue", "naive": "orange", "pedpred_only": "gray"}
    labels = {"enkf": "EnKF (Localized)", "cvn": "CrowdVarNet (ours)",
              "naive": "Naive (obs+prior)", "pedpred_only": "PedPred-only"}

    for ax, (key, title) in zip(axes.flat, channels):
        for method_key, result in all_results.items():
            steps = [m["step"] for m in result["per_step"]][skip:]
            vals = [m[key] for m in result["per_step"]][skip:]
            ax.plot(steps, vals, label=labels.get(method_key, method_key),
                    color=colors.get(method_key, "black"), linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_visualization(
    frame_idx: int,
    x_gt: np.ndarray,
    obs: np.ndarray,
    obs_mask: np.ndarray,
    results_dict: Dict[str, np.ndarray],
    save_path: Path,
):
    """Plot one frame: partial obs / GT / each method / density error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    n_methods = len(results_dict)
    fig = plt.figure(figsize=(5 * (n_methods + 2), 5))
    gs = gridspec.GridSpec(1, n_methods + 2, figure=fig, wspace=0.1)

    def _plot_state(ax, state, title, vmax=1.0):
        density = state[0].copy()
        vx, vy = state[1], state[2]
        speed = np.sqrt(vx**2 + vy**2)
        H, W = density.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        u = np.where(speed < 1e-6, np.nan, vx / (speed + 1e-9))
        v = np.where(speed < 1e-6, np.nan, vy / (speed + 1e-9))
        nan_mask = np.isnan(density)
        u[nan_mask] = np.nan
        v[nan_mask] = np.nan
        density[nan_mask] = 0.0
        im = ax.imshow(density, cmap="Blues", aspect="auto", origin="upper",
                       vmin=0, vmax=vmax)
        ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        return im

    # Partial obs
    ax0 = fig.add_subplot(gs[0])
    obs_vis = obs.copy()
    obs_vis[:, obs_mask[0] < 0.5] = np.nan
    _plot_state(ax0, obs_vis, f"Step {frame_idx}\nPartial Obs")

    # GT
    ax1 = fig.add_subplot(gs[1])
    _plot_state(ax1, x_gt, "Ground Truth")

    # Each method
    for i, (name, x_hat) in enumerate(results_dict.items()):
        ax = fig.add_subplot(gs[i + 2])
        _plot_state(ax, x_hat, name)

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Final comprehensive evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="benchmark/results/final")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--viz-frames", type=int, default=5, help="Number of frames to visualize")
    parser.add_argument("--methods", nargs="+",
                        default=["enkf", "cvn", "naive", "pedpred_only"])
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[final-eval] Loading data (split={args.split}, steps={args.num_steps})...", flush=True)
    frames = get_test_sequence(split=args.split, max_steps=args.num_steps)
    print(f"[final-eval] Loaded {len(frames)} frames", flush=True)

    print(f"[final-eval] Generating observations (seed={RANDOM_SEED})...", flush=True)
    observations = create_observation_sequence(frames, seed=RANDOM_SEED)

    # Build methods
    from .methods import (
        EnKFMethod, CrowdVarNetMethod, NaiveMethod, PedPredOnlyMethod,
    )
    method_map = {
        "enkf": lambda: EnKFMethod(device=args.device),
        "cvn": lambda: CrowdVarNetMethod(device=args.device),
        "naive": lambda: NaiveMethod(device=args.device),
        "pedpred_only": lambda: PedPredOnlyMethod(device=args.device),
    }

    all_results = {}
    all_predictions = {}  # store predictions for visualization

    for method_key in args.methods:
        print(f"\n{'='*60}", flush=True)
        print(f"[final-eval] Running: {method_key}", flush=True)
        print(f"{'='*60}", flush=True)

        method = method_map[method_key]()
        method.initialize(frames[0])

        per_step = []
        predictions = []
        for step_idx in range(len(frames)):
            obs_dict = observations[step_idx]
            t0 = time.time()
            x_hat = method.step(obs_dict, frames[step_idx])
            elapsed = time.time() - t0
            m = compute_all_metrics(x_hat, frames[step_idx], RHO_MASK_THR)
            m["time_s"] = elapsed
            m["step"] = step_idx
            per_step.append(m)
            predictions.append(x_hat.copy())

            if (step_idx + 1) % 25 == 0:
                print(f"  [{method.name}] step {step_idx+1}/{len(frames)}  "
                      f"RMSE={m['full_rmse']:.4f}  time={elapsed:.3f}s", flush=True)

        all_results[method_key] = {
            "name": method.name,
            "per_step": per_step,
        }
        all_predictions[method_key] = predictions

    # === Summary table ===
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE (support pixels, skip first 10 steps)")
    print(f"{'='*80}")
    skip = 10
    summary = {}
    header = f"{'Method':<25} {'RMSE':>8} {'ρ':>8} {'vx':>8} {'vy':>8} {'Time/step':>10}"
    print(header)
    print("-" * len(header))
    for k, v in all_results.items():
        steady = v["per_step"][skip:]
        s = {
            "full_rmse": float(np.mean([m["full_rmse"] for m in steady])),
            "rho": float(np.mean([m["rmse_support_rho"] for m in steady])),
            "vx": float(np.mean([m["rmse_support_vx"] for m in steady])),
            "vy": float(np.mean([m["rmse_support_vy"] for m in steady])),
            "time": float(np.mean([m["time_s"] for m in v["per_step"]])),
        }
        summary[k] = s
        print(f"{v['name']:<25} {s['full_rmse']:>8.4f} {s['rho']:>8.4f} "
              f"{s['vx']:>8.4f} {s['vy']:>8.4f} {s['time']:>9.4f}s")

    (save_dir / "summary_table.json").write_text(json.dumps(summary, indent=2))

    # === Per-step RMSE plot ===
    print(f"\n[final-eval] Generating per-step RMSE plot...", flush=True)
    plot_per_step_rmse(all_results, save_dir / "per_step_rmse.png", skip=5)

    # === Visualization frames ===
    print(f"[final-eval] Generating visualization frames...", flush=True)
    viz_dir = save_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Pick evenly spaced frames after warmup
    viz_indices = np.linspace(15, len(frames) - 1, args.viz_frames).astype(int)
    for idx in viz_indices:
        results_dict = {}
        for method_key in args.methods:
            name = all_results[method_key]["name"]
            results_dict[name] = all_predictions[method_key][idx]

        plot_visualization(
            frame_idx=idx,
            x_gt=frames[idx],
            obs=observations[idx]["obs_clean"],
            obs_mask=observations[idx]["obs_mask"],
            results_dict=results_dict,
            save_path=viz_dir / f"vis_step_{idx:03d}.png",
        )
        print(f"  Saved: vis_step_{idx:03d}.png")

    print(f"\n[final-eval] All done! Results in {save_dir}", flush=True)


if __name__ == "__main__":
    main()
