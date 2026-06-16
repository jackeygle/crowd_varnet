#!/bin/bash
#SBATCH --job-name=cvn_best_vis
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:15:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Run best model (v8 LSTM+Attn) for 300 steps, generate 300 visualizations.
set -euo pipefail

source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_cvn_root.sh
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_venv.sh
export PEDPRED_ATC_DATA_DIR=/scratch/work/zhangx29/data
source /scratch/work/zhangx29/crowd_varnet/sbatch/inc_atc_data_env.sh
export PYTHONPATH="${CVN_ROOT}:/scratch/work/zhangx29:/scratch/work/zhangx29/Partial_observation"
export PYTHONUNBUFFERED=1

PYTHON="${CVN_ROOT}/.venv/bin/python"

cd "${CVN_ROOT}"

${PYTHON} -c "
import sys, os, time, json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '${CVN_ROOT}')
from crowd_varnet.cli._common import build_model_from_ckpt
from benchmark.config import RANDOM_SEED, RHO_MASK_THR
from benchmark.data_loader import get_test_sequence
from benchmark.obs_generator import create_observation_sequence
from benchmark.metrics import compute_all_metrics

# Load best model
ckpt = '${CVN_ROOT}/runs/cvn_v11_4dvar_17944197/best.pt'
device = torch.device('cuda')
model, meta = build_model_from_ckpt(Path(ckpt), device=device)
model.eval()
T_hist = model.T_hist
print(f'Loaded: {ckpt}')

# Load data
frames = get_test_sequence(split='test', max_steps=300)
observations = create_observation_sequence(frames, seed=RANDOM_SEED)
print(f'Loaded {len(frames)} frames')

# Setup
save_dir = Path('benchmark/results/v11_vis')
viz_dir = save_dir / 'visualizations'
viz_dir.mkdir(parents=True, exist_ok=True)

# Initialize history
frame_t = torch.from_numpy(frames[0]).float().unsqueeze(0)
history_buf = frame_t.repeat(1, T_hist, 1, 1, 1).to(device)

# Run + visualize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

metrics_list = []
for step_idx in range(len(frames)):
    obs_dict = observations[step_idx]
    obs = torch.from_numpy(obs_dict['obs_clean']).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(obs_dict['obs_mask']).float().unsqueeze(0).to(device)

    saved = [(p, p.requires_grad) for p in model.parameters() if p.requires_grad]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        with torch.set_grad_enabled(True):
            x_hat = model.forward(history_buf, obs, mask)
    finally:
        for p, was in saved:
            p.requires_grad_(was)

    x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)
    history_buf = torch.cat([history_buf[:, 1:], x_hat.detach().unsqueeze(1)], dim=1)

    m = compute_all_metrics(x_hat_np, frames[step_idx], RHO_MASK_THR)
    metrics_list.append(m)

    # Visualization
    x_gt = frames[step_idx]
    obs_np = obs_dict['obs_clean']
    obs_mask_np = obs_dict['obs_mask']

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1)

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
        ax.imshow(density, cmap='Blues', aspect='auto', origin='upper', vmin=0, vmax=vmax)
        ax.quiver(x, y, u, v, color='black', scale=30, headwidth=3, headlength=4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Partial obs
    ax0 = fig.add_subplot(gs[0])
    obs_vis = obs_np.copy()
    obs_vis[:, obs_mask_np[0] < 0.5] = np.nan
    _plot_state(ax0, obs_vis, f'Step {step_idx} | Partial Obs')

    # GT
    ax1 = fig.add_subplot(gs[1])
    _plot_state(ax1, x_gt, 'Ground Truth')

    # Our prediction
    ax2 = fig.add_subplot(gs[2])
    _plot_state(ax2, x_hat_np, f'CrowdVarNet (RMSE={m[\"full_rmse\"]:.4f})')

    plt.tight_layout()
    fig.savefig(viz_dir / f'vis_{step_idx:04d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    if (step_idx + 1) % 50 == 0:
        print(f'  step {step_idx+1}/300  RMSE={m[\"full_rmse\"]:.4f}')

# Summary
skip = 10
steady = metrics_list[skip:]
summary = {
    'full_rmse': float(np.mean([m['full_rmse'] for m in steady])),
    'rho': float(np.mean([m['rmse_support_rho'] for m in steady])),
    'vx': float(np.mean([m['rmse_support_vx'] for m in steady])),
    'vy': float(np.mean([m['rmse_support_vy'] for m in steady])),
}
print(f'\\nRESULTS: RMSE={summary[\"full_rmse\"]:.4f} rho={summary[\"rho\"]:.4f} vx={summary[\"vx\"]:.4f} vy={summary[\"vy\"]:.4f}')
(save_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'Done! 300 visualizations in {viz_dir}')
"

echo "=== Done ==="
