#!/bin/bash
#SBATCH --job-name=cvn_v12_v15_cmp
#SBATCH --output=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.out
#SBATCH --error=/scratch/work/zhangx29/crowd_varnet/logs/%x_%j.err
#SBATCH --partition=gpu-debug
#SBATCH --time=00:25:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

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
import sys, os, json
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
    'v12': '${CVN_ROOT}/runs/cvn_v12_unobs_w_17945650/best.pt',
    'v13': '${CVN_ROOT}/runs/cvn_v13_unfreeze_17945681/best.pt',
    'v14': '${CVN_ROOT}/runs/cvn_v14_sched_17945714/best.pt',
    'v15': '${CVN_ROOT}/runs/cvn_v15_obs_token_17945773/best.pt',
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

save_dir = Path('benchmark/results/v12_v15_compare')
viz_dir = save_dir / 'visualizations'
viz_dir.mkdir(parents=True, exist_ok=True)

T_hist = list(models.values())[0].T_hist
history_bufs = {n: torch.from_numpy(frames[0]).float().unsqueeze(0).repeat(1, T_hist, 1, 1, 1).to(device) for n in models}
metrics_lists = {n: [] for n in models}

for step_idx in range(len(frames)):
    obs_dict = observations[step_idx]
    obs = torch.from_numpy(obs_dict['obs_clean']).float().unsqueeze(0).to(device)
    mask = torch.from_numpy(obs_dict['obs_mask']).float().unsqueeze(0).to(device)
    preds = {}
    for n, m in models.items():
        saved = [(p, p.requires_grad) for p in m.parameters() if p.requires_grad]
        for p, _ in saved: p.requires_grad_(False)
        try:
            with torch.set_grad_enabled(True):
                x_hat = m.forward(history_bufs[n], obs, mask)
        finally:
            for p, was in saved: p.requires_grad_(was)
        x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)
        history_bufs[n] = torch.cat([history_bufs[n][:, 1:], x_hat.detach().unsqueeze(1)], dim=1)
        preds[n] = x_hat_np
        metrics_lists[n].append(compute_all_metrics(x_hat_np, frames[step_idx], RHO_MASK_THR))

    n_methods = len(models)
    fig = plt.figure(figsize=(5 * (n_methods + 2), 5))
    gs = gridspec.GridSpec(1, n_methods + 2, figure=fig, wspace=0.1)
    def _plot(ax, state, title, vmax=1.0):
        density = state[0].copy()
        vx, vy = state[1], state[2]
        speed = np.sqrt(vx**2 + vy**2)
        H, W = density.shape
        x_g, y_g = np.meshgrid(np.arange(W), np.arange(H))
        nan_mask = np.isnan(density)
        density[nan_mask] = 0.0
        ax.imshow(density, cmap='Blues', aspect='auto', origin='upper', vmin=0, vmax=vmax)
        # Two-tier arrows:
        #   high density (>0.05): solid black arrows (model confident there are people)
        #   low density (0.01-0.05): gray semi-transparent (uncertain, but flow inferred)
        high_thr, low_thr = 0.05, 0.01
        # High density region
        valid_high = (density > high_thr) & (speed > 1e-6) & ~nan_mask
        u_h = np.where(valid_high, vx / (speed + 1e-9), np.nan)
        v_h = np.where(valid_high, vy / (speed + 1e-9), np.nan)
        ax.quiver(x_g, y_g, u_h, v_h, color='black', scale=30,
                  headwidth=3, headlength=4)
        # Low density region (gray, semi-transparent)
        valid_low = (density > low_thr) & (density <= high_thr) & (speed > 1e-6) & ~nan_mask
        u_l = np.where(valid_low, vx / (speed + 1e-9), np.nan)
        v_l = np.where(valid_low, vy / (speed + 1e-9), np.nan)
        ax.quiver(x_g, y_g, u_l, v_l, color='gray', alpha=0.5, scale=30,
                  headwidth=3, headlength=4)
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[0])
    obs_vis = obs_dict['obs_clean'].copy(); obs_vis[:, obs_dict['obs_mask'][0] < 0.5] = np.nan
    _plot(ax, obs_vis, f'Step {step_idx} | Obs')
    ax = fig.add_subplot(gs[1]); _plot(ax, frames[step_idx], 'GT')
    for i, n in enumerate(models):
        ax = fig.add_subplot(gs[i + 2]); _plot(ax, preds[n], n)
    plt.tight_layout()
    fig.savefig(viz_dir / f'vis_{step_idx:04d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    if (step_idx + 1) % 50 == 0:
        msg = f'  step {step_idx+1}/300  '
        for n, ml in metrics_lists.items():
            msg += f'{n}={ml[-1][\"full_rmse\"]:.4f} '
        print(msg)

skip = 10
summary = {}
print('\\nRESULTS (skip 10):')
for n, ml in metrics_lists.items():
    s = {
        'full': float(np.mean([m['full_rmse'] for m in ml[skip:]])),
        'rho': float(np.mean([m['rmse_support_rho'] for m in ml[skip:]])),
        'vx': float(np.mean([m['rmse_support_vx'] for m in ml[skip:]])),
        'vy': float(np.mean([m['rmse_support_vy'] for m in ml[skip:]])),
    }
    summary[n] = s
    print(f'  {n:<5} RMSE={s[\"full\"]:.4f} rho={s[\"rho\"]:.4f} vx={s[\"vx\"]:.4f} vy={s[\"vy\"]:.4f}')
(save_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
print(f'Done! 300 vis in {viz_dir}')
"
echo "=== Done ==="
