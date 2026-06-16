#!/bin/bash
#SBATCH --job-name=cvn_v12_v13_v14_vis
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# Compare v12 / v13 / v14 (epoch 1 best.pt) on 300 frames with viz.
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
import sys, os, json, time
import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, '${CVN_ROOT}')
from crowd_varnet.cli._common import build_model_from_ckpt
from benchmark.config import RANDOM_SEED, RHO_MASK_THR
from benchmark.data_loader import get_test_sequence
from benchmark.obs_generator import create_observation_sequence
from benchmark.metrics import compute_all_metrics

device = torch.device('cuda')
ckpts = {
    'v12 (unobs_w=5)': '${CVN_ROOT}/runs/cvn_v12_unobs_w_17945650/best.pt',
    'v13 (unfreeze)': '${CVN_ROOT}/runs/cvn_v13_unfreeze_17945681/best.pt',
    'v14 (sched_samp)': '${CVN_ROOT}/runs/cvn_v14_sched_17945714/best.pt',
}

models = {}
for name, p in ckpts.items():
    if not Path(p).exists():
        print(f'SKIP {name}: not found')
        continue
    m, _ = build_model_from_ckpt(Path(p), device=device)
    m.eval()
    models[name] = m
    print(f'Loaded: {name}')

frames = get_test_sequence(split='test', max_steps=300)
observations = create_observation_sequence(frames, seed=RANDOM_SEED)
print(f'Loaded {len(frames)} frames')

save_dir = Path('benchmark/results/v12_v14_compare')
viz_dir = save_dir / 'visualizations'
viz_dir.mkdir(parents=True, exist_ok=True)

T_hist = list(models.values())[0].T_hist

# Initialize history per model
history_bufs = {}
predictions = {n: [] for n in models}
metrics_lists = {n: [] for n in models}

frame_t = torch.from_numpy(frames[0]).float().unsqueeze(0)
for n in models:
    history_bufs[n] = frame_t.repeat(1, T_hist, 1, 1, 1).to(device)

for step_idx in range(len(frames)):
    obs_dict = observations[step_idx]
    obs = torch.from_numpy(obs_dict['obs_clean']).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(obs_dict['obs_mask']).float().unsqueeze(0).to(device)

    for n, m in models.items():
        saved = [(p, p.requires_grad) for p in m.parameters() if p.requires_grad]
        for p, _ in saved:
            p.requires_grad_(False)
        try:
            with torch.set_grad_enabled(True):
                x_hat = m.forward(history_bufs[n], obs, mask)
        finally:
            for p, was in saved:
                p.requires_grad_(was)

        x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)
        history_bufs[n] = torch.cat([history_bufs[n][:, 1:], x_hat.detach().unsqueeze(1)], dim=1)
        predictions[n].append(x_hat_np)
        mt = compute_all_metrics(x_hat_np, frames[step_idx], RHO_MASK_THR)
        metrics_lists[n].append(mt)

    # Visualization with quiver
    n_methods = len(models)
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
        ax.imshow(density, cmap='Blues', aspect='auto', origin='upper', vmin=0, vmax=vmax)
        ax.quiver(x, y, u, v, color='black', scale=30, headwidth=3, headlength=4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(gs[0])
    obs_vis = obs_dict['obs_clean'].copy()
    obs_vis[:, obs_dict['obs_mask'][0] < 0.5] = np.nan
    _plot_state(ax, obs_vis, f'Step {step_idx} | Obs')

    ax = fig.add_subplot(gs[1])
    _plot_state(ax, frames[step_idx], 'GT')

    for i, n in enumerate(models):
        ax = fig.add_subplot(gs[i + 2])
        _plot_state(ax, predictions[n][-1], n)

    plt.tight_layout()
    fig.savefig(viz_dir / f'vis_{step_idx:04d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    if (step_idx + 1) % 50 == 0:
        msg = f'  step {step_idx+1}/300  '
        for n, ml in metrics_lists.items():
            msg += f'{n.split()[0]}={ml[-1][\"full_rmse\"]:.4f} '
        print(msg)

# Summary
skip = 10
summary = {}
print('\nRESULTS (skip 10 steps):')
for n, ml in metrics_lists.items():
    steady = ml[skip:]
    s = {
        'full': float(np.mean([m['full_rmse'] for m in steady])),
        'rho': float(np.mean([m['rmse_support_rho'] for m in steady])),
        'vx': float(np.mean([m['rmse_support_vx'] for m in steady])),
        'vy': float(np.mean([m['rmse_support_vy'] for m in steady])),
    }
    summary[n] = s
    print(f'  {n:<25} RMSE={s[\"full\"]:.4f} rho={s[\"rho\"]:.4f} vx={s[\"vx\"]:.4f} vy={s[\"vy\"]:.4f}')

(save_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'Done! 300 vis in {viz_dir}')
"

echo "=== Done ==="
