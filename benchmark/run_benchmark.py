"""
实验 B：状态估计对比 — 主实验脚本。

在同一测试序列、同一观测几何下，对比所有方法的全场重建精度。

用法:
    python -m benchmark.run_benchmark --device cuda --num-steps 300 --save-dir benchmark/results/exp_b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import NUM_STEPS, WARMUP_STEPS, RHO_MASK_THR, RANDOM_SEED, CH_NAMES
from .data_loader import get_test_sequence
from .obs_generator import create_observation_sequence
from .metrics import compute_all_metrics


def run_single_method(method, frames, observations, warmup: int = WARMUP_STEPS):
    """Run a single method on the full sequence. Returns per-step metrics + timing."""
    # Initialize with first frame
    method.initialize(frames[0])

    # Warmup: feed GT frames into history (for methods that need it)
    # For EnKF: the ensemble is initialized from first frame
    # For CrowdVarNet: history buffer is filled with first frame (then updated)

    per_step_metrics = []
    per_step_time = []

    for step_idx in range(len(frames)):
        obs_dict = observations[step_idx]
        true_frame = frames[step_idx]

        t0 = time.time()
        x_hat = method.step(obs_dict, true_frame)
        elapsed = time.time() - t0
        per_step_time.append(elapsed)

        # Compute metrics
        metrics = compute_all_metrics(x_hat, true_frame, rho_thr=RHO_MASK_THR)
        metrics["step"] = step_idx
        metrics["time_s"] = elapsed
        per_step_metrics.append(metrics)

        if (step_idx + 1) % 50 == 0 or step_idx == 0:
            print(
                f"  [{method.name}] step {step_idx + 1}/{len(frames)}  "
                f"RMSE={metrics['full_rmse']:.4f}  "
                f"rho={metrics['rmse_support_rho']:.4f}  "
                f"vx={metrics['rmse_support_vx']:.4f}  "
                f"time={elapsed:.3f}s",
                flush=True,
            )

    return per_step_metrics, per_step_time


def summarize_metrics(per_step_metrics: List[Dict], skip_transient: int = 20) -> Dict[str, float]:
    """Compute summary statistics (mean over steady-state steps)."""
    steady = per_step_metrics[skip_transient:]
    if not steady:
        steady = per_step_metrics

    summary = {}
    # Mean of each metric key
    keys = [k for k in steady[0].keys() if k not in ("step", "time_s")]
    for k in keys:
        vals = [m[k] for m in steady]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"] = float(np.std(vals))

    # Timing
    all_times = [m["time_s"] for m in per_step_metrics]
    summary["mean_time_per_step"] = float(np.mean(all_times))
    summary["total_time"] = float(np.sum(all_times))
    summary["num_steps"] = len(per_step_metrics)
    summary["skip_transient"] = skip_transient

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark: state estimation comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--save-dir", type=str, default="benchmark/results/exp_b")
    parser.add_argument("--methods", nargs="+",
                        default=["enkf", "pf", "cvn", "naive", "pedpred_only"],
                        choices=["enkf", "pf", "cvn", "naive", "pedpred_only"])
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--split", type=str, default="test2")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] Loading test data (split={args.split}, steps={args.num_steps})...", flush=True)
    frames = get_test_sequence(split=args.split, max_steps=args.num_steps)
    print(f"[benchmark] Loaded {len(frames)} frames, shape={frames[0].shape}", flush=True)

    print(f"[benchmark] Generating observation sequence (seed={args.seed})...", flush=True)
    observations = create_observation_sequence(frames, seed=args.seed)
    print(f"[benchmark] Generated {len(observations)} observation steps", flush=True)

    # Build methods
    from .methods import (
        EnKFMethod, ParticleFilterMethod, CrowdVarNetMethod,
        NaiveMethod, PedPredOnlyMethod,
    )

    method_map = {
        "enkf": lambda: EnKFMethod(device=args.device),
        "pf": lambda: ParticleFilterMethod(device=args.device),
        "cvn": lambda: CrowdVarNetMethod(device=args.device),
        "naive": lambda: NaiveMethod(device=args.device),
        "pedpred_only": lambda: PedPredOnlyMethod(device=args.device),
    }

    all_results = {}
    for method_key in args.methods:
        print(f"\n{'='*60}", flush=True)
        print(f"[benchmark] Running: {method_key}", flush=True)
        print(f"{'='*60}", flush=True)

        method = method_map[method_key]()
        per_step, times = run_single_method(method, frames, observations)
        summary = summarize_metrics(per_step)

        all_results[method_key] = {
            "name": method.name,
            "summary": summary,
            "per_step": per_step,
        }

        # Save per-method results
        method_file = save_dir / f"{method_key}_results.json"
        method_file.write_text(json.dumps(all_results[method_key], indent=2), encoding="utf-8")
        print(f"\n[{method.name}] Summary:", flush=True)
        print(f"  mean_full_rmse = {summary['mean_full_rmse']:.4f} ± {summary['std_full_rmse']:.4f}")
        print(f"  mean_rmse_support_rho = {summary['mean_rmse_support_rho']:.4f}")
        print(f"  mean_rmse_support_vx  = {summary['mean_rmse_support_vx']:.4f}")
        print(f"  mean_rmse_support_vy  = {summary['mean_rmse_support_vy']:.4f}")
        print(f"  mean_time_per_step    = {summary['mean_time_per_step']:.4f}s")

    # Save combined summary table
    summary_table = {k: v["summary"] for k, v in all_results.items()}
    (save_dir / "summary_table.json").write_text(
        json.dumps(summary_table, indent=2), encoding="utf-8"
    )

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE (steady-state mean, skip first 20 steps)")
    print(f"{'='*80}")
    header = f"{'Method':<25} {'RMSE':>8} {'ρ_RMSE':>8} {'vx_RMSE':>8} {'vy_RMSE':>8} {'Time/step':>10}"
    print(header)
    print("-" * len(header))
    for k, v in all_results.items():
        s = v["summary"]
        print(
            f"{v['name']:<25} "
            f"{s['mean_full_rmse']:>8.4f} "
            f"{s['mean_rmse_support_rho']:>8.4f} "
            f"{s['mean_rmse_support_vx']:>8.4f} "
            f"{s['mean_rmse_support_vy']:>8.4f} "
            f"{s['mean_time_per_step']:>9.4f}s"
        )

    print(f"\n[benchmark] Results saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
