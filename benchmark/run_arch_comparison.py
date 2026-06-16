"""
架构对比评测：v7 (GRU) vs v8 (LSTM) vs 他们教师 vs EnKF。

输出:
  - summary_table.json
  - per_step_rmse.png
  - visualizations/ (300 帧)

用法:
    python -m benchmark.run_arch_comparison --device cuda --num-steps 300 \
        --save-dir benchmark/results/arch_comparison
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
    PROJECT_ROOT, RANDOM_SEED, RHO_MASK_THR,
    WARMUP_STEPS, PO_ROOT, THEIR_TEACHER_CKPT,
)
from .data_loader import get_test_sequence
from .obs_generator import create_observation_sequence
from .metrics import compute_all_metrics


# ============================================================
# Single-model CrowdVarNet wrapper (loads one best.pt)
# ============================================================
class SingleCrowdVarNetMethod:
    def __init__(self, name: str, ckpt_path: str, device: str = "cuda"):
        self.name = name
        cvn_root = str(PROJECT_ROOT)
        if cvn_root not in sys.path:
            sys.path.insert(0, cvn_root)
        from crowd_varnet.cli._common import build_model_from_ckpt

        self.device = torch.device(device)
        self.model, self.meta = build_model_from_ckpt(
            Path(ckpt_path), device=self.device
        )
        self.model.eval()
        self.T_hist = int(self.model.T_hist)
        self.history_buf = None

    def initialize(self, first_frame: np.ndarray):
        frame_t = torch.from_numpy(first_frame).float().unsqueeze(0)
        self.history_buf = frame_t.repeat(1, self.T_hist, 1, 1, 1).to(self.device)

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        obs_mask = torch.from_numpy(obs_dict["obs_mask"]).float().unsqueeze(0).to(self.device)
        obs = torch.from_numpy(obs_dict["obs_clean"]).float().unsqueeze(0).to(self.device)

        saved = [(p, p.requires_grad) for p in self.model.parameters() if p.requires_grad]
        for p, _ in saved:
            p.requires_grad_(False)
        try:
            with torch.set_grad_enabled(True):
                x_hat = self.model.forward(self.history_buf, obs, obs_mask)
        finally:
            for p, was in saved:
                p.requires_grad_(was)

        x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)
        self.history_buf = torch.cat(
            [self.history_buf[:, 1:], x_hat.detach().unsqueeze(1)], dim=1
        )
        return x_hat_np


def plot_per_step(all_results: Dict, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    channels = [("full_rmse", "Full RMSE (support)"),
                ("rmse_support_rho", "Density ρ RMSE"),
                ("rmse_support_vx", "Velocity vx RMSE"),
                ("rmse_support_vy", "Velocity vy RMSE")]

    colors = ["red", "blue", "green", "purple", "orange"]
    skip = 10

    for ax, (key, title) in zip(axes.flat, channels):
        for i, (method_name, result) in enumerate(all_results.items()):
            steps = [m["step"] for m in result["per_step"]][skip:]
            vals = [m[key] for m in result["per_step"]][skip:]
            ax.plot(steps, vals, label=method_name,
                    color=colors[i % len(colors)], linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("RMSE")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_frame(step_idx, x_gt, obs, obs_mask, results_dict, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_methods = len(results_dict)
    fig, axes = plt.subplots(1, n_methods + 2, figsize=(3.5 * (n_methods + 2), 3.5))

    def _plot(ax, data, title):
        im = ax.imshow(data, cmap="Blues", aspect="auto", origin="upper", vmin=0, vmax=1.0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # GT
    _plot(axes[0], x_gt[0], f"GT (step {step_idx})")

    # Obs
    obs_vis = obs[0].copy()
    obs_vis[obs_mask[0] < 0.5] = np.nan
    _plot(axes[1], obs_vis, "Partial Obs")

    # Methods
    for i, (name, pred) in enumerate(results_dict.items()):
        _plot(axes[i + 2], pred[0], name)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--save-dir", type=str, default="benchmark/results/arch_comparison")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--no-enkf", action="store_true", help="Skip EnKF (slow)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = save_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Discover checkpoints
    v7_ckpt = str(PROJECT_ROOT / "runs/cvn_ensemble_v7_seed0_17932380/best.pt")
    v8_ckpt = str(PROJECT_ROOT / "runs/cvn_ensemble_v8_seed0_17932480/best.pt")
    their_ckpt = str(PROJECT_ROOT / "runs/cvn_their_teacher_v7_17933090/best.pt")

    # Build methods
    methods = {}

    if Path(v7_ckpt).exists():
        methods["CrowdVarNet-v7 (GRU+Attn)"] = SingleCrowdVarNetMethod(
            "CrowdVarNet-v7 (GRU+Attn)", v7_ckpt, args.device
        )
        print(f"  Loaded v7: {v7_ckpt}")

    if Path(v8_ckpt).exists():
        methods["CrowdVarNet-v8 (LSTM+Attn)"] = SingleCrowdVarNetMethod(
            "CrowdVarNet-v8 (LSTM+Attn)", v8_ckpt, args.device
        )
        print(f"  Loaded v8: {v8_ckpt}")

    if Path(their_ckpt).exists():
        # Their teacher uses TheirPedPredAdapter — need special loading
        cvn_root = str(PROJECT_ROOT)
        if cvn_root not in sys.path:
            sys.path.insert(0, cvn_root)
        from crowd_varnet.cli._common import load_training_meta
        from crowd_varnet.models.their_teacher_adapter import TheirPedPredAdapter
        from crowd_varnet.models import CrowdVarNet

        their_meta = load_training_meta(Path(their_ckpt).parent)
        their_device = torch.device(args.device)
        ped_their = TheirPedPredAdapter(
            str(THEIR_TEACHER_CKPT), device=str(their_device)
        )
        their_model = CrowdVarNet(
            ped_pred=ped_their,
            freeze_phi=True,
            T_hist=int(their_meta.get("nin", 5)),
            n_iter=int(their_meta.get("n_iter", 8)),
            w_prior=float(their_meta.get("w_prior", 0.5)),
            ch_weights=tuple(float(x) for x in their_meta.get("ch_weights", [3,1.5,1,0])),
            rho_mask_thr=float(their_meta.get("rho_mask_thr", 0.05)),
            solver_hidden=int(their_meta.get("solver_hidden", 256)),
            solver_kernel=int(their_meta.get("solver_kernel", 3)),
            solver_share=True,
            solver_dropout=float(their_meta.get("solver_dropout", 0.0)),
            predict_uncertainty=bool(their_meta.get("predict_uncertainty", False)),
            solver_use_attention=True,
            solver_attn_heads=4,
            solver_momentum=0.0,
            solver_rnn_type="lstm",
        ).to(their_device)
        their_payload = torch.load(their_ckpt, map_location=their_device, weights_only=False)
        their_sd = their_payload["model_state_dict"] if "model_state_dict" in their_payload else their_payload
        # Remap old key names
        their_sd = {k.replace("solver.convgru.", "solver.rnn_cell."): v for k, v in their_sd.items()}
        their_model.load_state_dict(their_sd, strict=False)
        their_model.eval()

        class _TheirTeacherWrapper:
            name = "CrowdVarNet+TheirTeacher"
            def __init__(self, model, device):
                self.model = model
                self.device = device
                self.T_hist = model.T_hist
                self.history_buf = None
            def initialize(self, first_frame):
                frame_t = torch.from_numpy(first_frame).float().unsqueeze(0)
                self.history_buf = frame_t.repeat(1, self.T_hist, 1, 1, 1).to(self.device)
            def step(self, obs_dict, true_frame=None):
                obs_mask = torch.from_numpy(obs_dict["obs_mask"]).float().unsqueeze(0).to(self.device)
                obs = torch.from_numpy(obs_dict["obs_clean"]).float().unsqueeze(0).to(self.device)
                saved = [(p, p.requires_grad) for p in self.model.parameters() if p.requires_grad]
                for p, _ in saved:
                    p.requires_grad_(False)
                try:
                    with torch.set_grad_enabled(True):
                        x_hat = self.model.forward(self.history_buf, obs, obs_mask)
                finally:
                    for p, was in saved:
                        p.requires_grad_(was)
                x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)
                self.history_buf = torch.cat(
                    [self.history_buf[:, 1:], x_hat.detach().unsqueeze(1)], dim=1
                )
                return x_hat_np

        methods["CrowdVarNet+TheirTeacher"] = _TheirTeacherWrapper(their_model, their_device)
        print(f"  Loaded their teacher: {their_ckpt}")

    if not args.no_enkf:
        from .methods import EnKFMethod
        enkf = EnKFMethod(device="cpu")
        methods["EnKF"] = enkf
        print("  Loaded EnKF")

    if not methods:
        print("ERROR: No methods found!")
        sys.exit(1)

    # Load data
    print(f"\n[arch-eval] Loading data (split={args.split}, steps={args.num_steps})...")
    frames = get_test_sequence(split=args.split, max_steps=args.num_steps)
    print(f"[arch-eval] Loaded {len(frames)} frames")

    print(f"[arch-eval] Generating observations (seed={RANDOM_SEED})...")
    observations = create_observation_sequence(frames, seed=RANDOM_SEED)

    # Run each method
    all_results = {}
    all_predictions = {}

    for method_name, method in methods.items():
        print(f"\n{'='*60}")
        print(f"  Running: {method_name}")
        print(f"{'='*60}")

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

            if (step_idx + 1) % 50 == 0:
                print(f"    step {step_idx+1}/{len(frames)}  "
                      f"RMSE={m['full_rmse']:.4f}  time={elapsed:.3f}s")

        all_results[method_name] = {"per_step": per_step}
        all_predictions[method_name] = predictions

    # Summary table
    print(f"\n{'='*80}")
    print("ARCHITECTURE COMPARISON (support pixels, skip first 10 steps)")
    print(f"{'='*80}")
    skip = 10
    summary = {}
    header = f"{'Method':<30} {'RMSE':>8} {'ρ':>8} {'vx':>8} {'vy':>8} {'Time':>8}"
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
        print(f"{k:<30} {s['full_rmse']:>8.4f} {s['rho']:>8.4f} "
              f"{s['vx']:>8.4f} {s['vy']:>8.4f} {s['time']:>7.3f}s")

    (save_dir / "summary_table.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Saved: {save_dir / 'summary_table.json'}")

    # Per-step plot
    plot_per_step(all_results, save_dir / "per_step_rmse.png")

    # Visualizations (every frame)
    print(f"\n[arch-eval] Generating {len(frames)} visualizations...")
    for step_idx in range(len(frames)):
        results_dict = {}
        for method_name in all_results:
            results_dict[method_name] = all_predictions[method_name][step_idx]
        plot_frame(
            step_idx, frames[step_idx],
            observations[step_idx]["obs_clean"],
            observations[step_idx]["obs_mask"],
            results_dict,
            viz_dir / f"vis_{step_idx:04d}.png",
        )
        if (step_idx + 1) % 50 == 0:
            print(f"    vis {step_idx+1}/{len(frames)}")

    print(f"\n[arch-eval] Done! Results in {save_dir}")


if __name__ == "__main__":
    main()
