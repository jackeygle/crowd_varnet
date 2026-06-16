"""
实验 A：Teacher 模型对比 — 纯预测精度。

对比两个 PedPred 模型在开环预测上的精度（单步 + 多步 rollout）。

用法:
    python -m benchmark.run_teacher_compare --device cuda --save-dir benchmark/results/exp_a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import (
    GRID_SIZE, STATE_SHAPE, RHO_MASK_THR,
    OUR_TEACHER_CKPT, OUR_TEACHER_ARCH,
    THEIR_TEACHER_CKPT, PO_ROOT,
    WARMUP_STEPS,
)
from .data_loader import get_test_sequence
from .metrics import compute_all_metrics, per_channel_rmse, density_support_mask


def load_our_teacher(device: str):
    """Load our PedPred3_gru_mid teacher."""
    import os
    # Allow env var override for testing different teacher versions
    ckpt_path = os.environ.get("CVN_OUR_TEACHER_CKPT", str(OUR_TEACHER_CKPT))

    cvn_root = str(Path(ckpt_path).parent.parent.parent)
    if cvn_root not in sys.path:
        sys.path.insert(0, cvn_root)

    from crowd_varnet.models import load_frozen_pedpred
    from crowd_varnet.models.prior import FrozenPedPredPrior

    dev = torch.device(device)
    ped = load_frozen_pedpred(ckpt_path, dev, arch=OUR_TEACHER_ARCH)
    prior = FrozenPedPredPrior(ped, freeze=True).to(dev).eval()
    print(f"  [our teacher] loaded: {ckpt_path}", flush=True)
    return prior, dev


def load_their_teacher(device: str):
    """Load their PedPred3 teacher."""
    po_parent = str(PO_ROOT.parent)
    if po_parent not in sys.path:
        sys.path.insert(0, po_parent)

    from Partial_observation.utils import load_model
    # Load on CPU to avoid device mismatch in their GRU hidden state init
    model = load_model(str(THEIR_TEACHER_CKPT), "cpu")
    return model, torch.device("cpu")


def predict_one_step_ours(prior, history_buf, device):
    """Our teacher: [1, T, 4, H, W] -> [4, H, W]."""
    with torch.no_grad():
        x_pred = prior(history_buf)
    return x_pred[0].cpu().numpy().astype(np.float64)


def predict_one_step_theirs(model, frame, device):
    """Their teacher: single frame [1, 1, 4, H, W] -> [4, H, W]."""
    model_device = next(model.parameters()).device
    x_in = torch.from_numpy(frame).float().reshape(1, 1, *STATE_SHAPE).to(model_device)
    with torch.no_grad():
        out = model(x_in, horizon=1)
    # Their model returns GridData; extract tensor
    if hasattr(out, 'as_tensor'):
        out_t = out.as_tensor('density', 'vel_mean', 'vel_var')
    else:
        out_t = out
    if out_t.dim() == 5:
        out_t = out_t[:, 0]  # [B, T, C, H, W] -> [B, C, H, W]
    return out_t[0].detach().cpu().numpy().astype(np.float64)


def run_single_step_eval(frames: List[np.ndarray], device: str) -> Dict:
    """Single-step prediction: given GT frame t, predict frame t+1."""
    our_prior, our_dev = load_our_teacher(device)
    their_model, their_dev = load_their_teacher(device)

    our_metrics = []
    their_metrics = []

    T_hist = WARMUP_STEPS
    num_pairs = len(frames) - 1

    for t in range(num_pairs):
        gt_next = frames[t + 1]

        # Our teacher: needs T_hist frames of history
        if t < T_hist:
            # Pad with first frame
            hist_frames = [frames[0]] * (T_hist - t - 1) + frames[:t + 1]
        else:
            hist_frames = frames[t - T_hist + 1:t + 1]
        history_buf = torch.from_numpy(
            np.stack(hist_frames, axis=0)
        ).float().unsqueeze(0).to(our_dev)  # [1, T, 4, H, W]
        our_pred = predict_one_step_ours(our_prior, history_buf, our_dev)

        # Their teacher: single frame input
        their_pred = predict_one_step_theirs(their_model, frames[t], their_dev)

        # Metrics
        our_m = compute_all_metrics(our_pred, gt_next, RHO_MASK_THR)
        their_m = compute_all_metrics(their_pred, gt_next, RHO_MASK_THR)
        our_metrics.append(our_m)
        their_metrics.append(their_m)

        if (t + 1) % 100 == 0:
            print(f"  single-step eval: {t + 1}/{num_pairs}", flush=True)

    return {"ours": our_metrics, "theirs": their_metrics}


def run_rollout_eval(frames: List[np.ndarray], device: str, rollout_len: int = 100) -> Dict:
    """Multi-step open-loop rollout: autoregressive prediction without observations.

    NOTE: This is a deliberately challenging test. In real use, both teachers
    are paired with state estimation methods (EnKF / CrowdVarNet) that
    re-anchor predictions with observations every step. Open-loop rollout
    over many steps reflects pure teacher extrapolation capability, which
    is NOT how either teacher is used in practice.

    Tests rollout at multiple horizons: 1, 3, 5, 10, 20, 50 steps.
    Reports per-step metrics across multiple starting points (averaged) for
    statistical robustness.
    """
    our_prior, our_dev = load_our_teacher(device)
    their_model, their_dev = load_their_teacher(device)

    T_hist = WARMUP_STEPS
    # Run multiple rollouts from different starting points and average
    # Each rollout is `rollout_len` steps long
    num_starts = max(1, (len(frames) - T_hist - rollout_len) // rollout_len)

    # Per-step metrics aggregated across multiple starting points
    our_per_step = [[] for _ in range(rollout_len)]
    their_per_step = [[] for _ in range(rollout_len)]

    print(f"  averaging across {num_starts} rollout windows of {rollout_len} steps", flush=True)

    for start_idx in range(num_starts):
        start = T_hist + start_idx * rollout_len
        if start + rollout_len > len(frames):
            break

        # Initialize history from GT
        our_history = torch.from_numpy(
            np.stack(frames[start - T_hist:start], axis=0)
        ).float().unsqueeze(0).to(our_dev)
        their_last_frame = frames[start - 1].copy()

        for t in range(rollout_len):
            gt_frame = frames[start + t]

            our_pred = predict_one_step_ours(our_prior, our_history, our_dev)
            pred_t = torch.from_numpy(our_pred).float().unsqueeze(0).unsqueeze(0).to(our_dev)
            our_history = torch.cat([our_history[:, 1:], pred_t], dim=1)

            their_pred = predict_one_step_theirs(their_model, their_last_frame, their_dev)
            their_last_frame = their_pred.copy()

            our_m = compute_all_metrics(our_pred, gt_frame, RHO_MASK_THR)
            their_m = compute_all_metrics(their_pred, gt_frame, RHO_MASK_THR)
            our_per_step[t].append(our_m)
            their_per_step[t].append(their_m)

        if (start_idx + 1) % 1 == 0:
            ours_rmse = np.mean([m["full_rmse"] for m in our_per_step[rollout_len - 1]])
            theirs_rmse = np.mean([m["full_rmse"] for m in their_per_step[rollout_len - 1]])
            print(
                f"  rollout window {start_idx + 1}/{num_starts}  "
                f"final ours_rmse={ours_rmse:.4f}  theirs_rmse={theirs_rmse:.4f}",
                flush=True,
            )

    # Aggregate: mean per step across windows
    def _agg(per_step_list):
        out = []
        for t_list in per_step_list:
            if not t_list:
                continue
            avg = {}
            for k in t_list[0].keys():
                if k == "step":
                    continue
                vals = [m[k] for m in t_list]
                avg[k] = float(np.mean(vals))
            avg["step"] = len(out)
            out.append(avg)
        return out

    our_rollout_metrics = _agg(our_per_step)
    their_rollout_metrics = _agg(their_per_step)
    return {
        "ours": our_rollout_metrics,
        "theirs": their_rollout_metrics,
        "num_windows": num_starts,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark: teacher model comparison")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--rollout-len", type=int, default=100)
    parser.add_argument("--save-dir", type=str, default="benchmark/results/exp_a")
    parser.add_argument("--split", type=str, default="test2")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[teacher-compare] Loading test data...", flush=True)
    frames = get_test_sequence(split=args.split, max_steps=args.num_steps)
    print(f"[teacher-compare] Loaded {len(frames)} frames", flush=True)

    # Experiment A1: Single-step prediction
    print(f"\n{'='*60}")
    print("[Exp A1] Single-step prediction comparison")
    print(f"{'='*60}", flush=True)
    single_step = run_single_step_eval(frames, args.device)

    # Summarize
    for who in ("ours", "theirs"):
        metrics_list = single_step[who]
        mean_rmse = np.mean([m["full_rmse"] for m in metrics_list])
        mean_rho = np.mean([m["rmse_support_rho"] for m in metrics_list])
        mean_vx = np.mean([m["rmse_support_vx"] for m in metrics_list])
        mean_vy = np.mean([m["rmse_support_vy"] for m in metrics_list])
        print(
            f"  {who:>6s}: RMSE={mean_rmse:.4f}  "
            f"ρ={mean_rho:.4f}  vx={mean_vx:.4f}  vy={mean_vy:.4f}"
        )

    # Experiment A2: Multi-step rollout
    print(f"\n{'='*60}")
    print(f"[Exp A2] Multi-step rollout (len={args.rollout_len})")
    print(f"{'='*60}", flush=True)
    rollout = run_rollout_eval(frames, args.device, rollout_len=args.rollout_len)

    for who in ("ours", "theirs"):
        metrics_list = rollout[who]
        # Report at multiple horizons
        horizons = [0, 2, 4, 9, 19, 49]  # steps 1, 3, 5, 10, 20, 50
        horizons = [h for h in horizons if h < len(metrics_list)]
        rmse_at = [metrics_list[h]["full_rmse"] for h in horizons]
        labels = [f"t={h+1}" for h in horizons]
        print(f"  {who:>6s}: " + "  ".join(
            f"{l}={r:.4f}" for l, r in zip(labels, rmse_at)
        ))
        mean_short = np.mean([m["full_rmse"] for m in metrics_list[:5]])  # mean over first 5 steps
        print(f"          mean(t=1..5)={mean_short:.4f}")

    # Save results
    results = {
        "single_step": single_step,
        "rollout": rollout,
        "config": {
            "our_ckpt": str(OUR_TEACHER_CKPT),
            "their_ckpt": str(THEIR_TEACHER_CKPT),
            "num_frames": len(frames),
            "rollout_len": args.rollout_len,
            "split": args.split,
        },
    }
    (save_dir / "teacher_compare_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\n[teacher-compare] Results saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
