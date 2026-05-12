"""
输出 PNG 与 ``crowd_varnet.deps.utils_plot.save_step_plot`` 同款布局（四列 × Blues+quiver；
第四列为密度 |Pred−GT|，因无集合 spread）。文件名为 ``step_%03d.png``。

用法::

    cd /scratch/work/zhangx29/crowd_varnet
    export PYTHONPATH=/scratch/work/zhangx29/crowd_varnet
    export PEDPRED_BATCH=16 PEDPRED_NIN=5 PEDPRED_NOUT=1 \\
           PEDPRED_RESOLUTION=1.0 PEDPRED_PERIOD=1.0 PEDPRED_KERNEL=tri

    python -m crowd_varnet.infer_cli \\
      --ckpt /path/to/best.pt \\
      --out-dir /path/to/infer_vis \\
      --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .deps.dataset_atc import get_atc_data
from .deps.utils_plot import plot_generated_matrix_on_ax

from .core import CrowdVarNet, load_frozen_pedpred  # noqa: E402
from .train_cli import wrap_loader_varnet  # noqa: E402


def _partial_obs_as_nan(
    obs_4hw: np.ndarray,
    mask_hw: np.ndarray,
) -> np.ndarray:
    """与 ENKF 一致：未观测格点设为 NaN（utils.plot_generated_matrix_on_ax 用 NaN 隐藏箭头）。"""
    out = obs_4hw.astype(np.float64, copy=True)
    m = mask_hw > 0.5
    for c in range(4):
        out[c][~m] = np.nan
    return out


@torch.no_grad()
def _save_step_plot_project_style(
    partial_obs: np.ndarray,
    true_state: np.ndarray,
    estimated_mean: np.ndarray,
    step_idx: int,
    out_dir: Path,
) -> None:
    """
    与 ``crowd_varnet.deps.utils_plot.save_step_plot`` 同款布局：
    Partial Obs / True / Estimated Mean / 第四列 Reds。

    无集合 spread 时，第四列画密度绝对误差 |ρ̂−ρ|（形状与 ENKF 的 σρ 面板一致）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    spread = np.zeros_like(true_state, dtype=np.float64)
    spread[0] = np.abs(estimated_mean[0].astype(np.float64) - true_state[0].astype(np.float64))
    spread[np.isnan(spread)] = 0.0
    rho_e = spread[0]
    emean = float(np.mean(rho_e))
    elocmax = float(np.max(rho_e))

    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15)

    ax0 = fig.add_subplot(gs[0])
    im0 = plot_generated_matrix_on_ax(partial_obs, ax0)
    ax0.set_title(f"Step {step_idx} - Partial Obs")

    ax1 = fig.add_subplot(gs[1])
    im1 = plot_generated_matrix_on_ax(true_state, ax1)
    ax1.set_title("True State")

    ax2 = fig.add_subplot(gs[2])
    im2 = plot_generated_matrix_on_ax(estimated_mean, ax2)
    ax2.set_title("Estimated Mean")

    ax3 = fig.add_subplot(gs[3])
    im3 = ax3.imshow(rho_e, cmap="Reds", origin="upper", aspect="auto", vmin=0.0)
    ax3.set_title(f"Density |Pred−GT|\nmean={emean:.4f}  max={elocmax:.4f}")
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"step_{step_idx:03d}.png"
    plt.savefig(filename)
    plt.close(fig)


def _load_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "training_meta.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrowdVarNet inference + viz from best.pt")
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to best.pt (or last.pt); training_meta.json 应在同目录)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, required=True, help="Directory for PNG + metrics.json")
    p.add_argument("--val-workers", type=int, default=0)
    p.add_argument(
        "--batch",
        type=int,
        default=None,
        help="验证 DataLoader batch（默认 training_meta.json 的 batch 或 PEDPRED_BATCH）",
    )
    p.add_argument("--max-batches", type=int, default=None, help="Limit val batches (default: full val)")
    p.add_argument(
        "--viz-n",
        type=int,
        default=4,
        help="保存前 N 个验证样本的可视化 step_{000..N-1}.png（跨 batch；与 ENKF 同款四列）",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="每 N 个 val batch 打印一行进度（1=每个 batch；0=仅首尾与最终汇总）",
    )
    p.add_argument(
        "--rho-mask-thr",
        type=float,
        default=None,
        help="覆盖 training_meta 的 rho_mask_thr；默认读 meta（缺省为 0.05）",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    return args


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    ckpt_path = Path(args.ckpt).resolve()
    run_dir = ckpt_path.parent
    meta = _load_meta(run_dir)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    default_nin = int(os.environ.get("PEDPRED_NIN", "5"))
    nin = int(meta.get("nin", default_nin))
    default_batch = int(os.environ.get("PEDPRED_BATCH", "16"))
    batch = int(meta.get("batch", default_batch)) if args.batch is None else int(args.batch)

    obs_mode = str(meta.get("obs_mode", "sensor"))
    sensing_range = float(meta.get("sensing_range", 5.0))
    num_agents = int(meta.get("num_agents", 3))
    n_iter = int(meta.get("n_iter", 8))
    w_prior = float(meta.get("w_prior", 0.5))
    rho_mask_thr = (
        float(args.rho_mask_thr)
        if args.rho_mask_thr is not None
        else float(meta.get("rho_mask_thr", 0.05))
    )
    arch = str(meta.get("arch", "pedpred3"))
    ped_path = meta.get("pedpred_ckpt")
    if not ped_path:
        raise SystemExit("training_meta.json 缺少 pedpred_ckpt，无法加载 PedPred")
    ped_path = str(Path(ped_path).resolve())

    nout = int(meta.get("nout", os.environ.get("PEDPRED_NOUT", "1")))
    val_loader = get_atc_data(
        "valid",
        batch=batch,
        nin=nin,
        nout=nout,
        num_workers=args.val_workers,
        drop_last=False,
    )
    val_loader = wrap_loader_varnet(
        val_loader,
        seed=1,
        obs_mode=obs_mode,
        partial_frac=0.35,
        sensing_range=sensing_range,
        num_agents=num_agents,
        batch_size=batch,
        num_workers=args.val_workers,
        shuffle=False,
        drop_last=False,
    )

    ped = load_frozen_pedpred(ped_path, device, arch=arch)
    payload = torch.load(ckpt_path, map_location=device)
    # 自动检测 checkpoint 是否用 GRU 训练
    use_gru = any("gru" in k for k in payload["model_state_dict"].keys())
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=nin,
        n_iter=n_iter,
        use_gru=use_gru,
        w_prior=w_prior,
        rho_mask_thr=rho_mask_thr,
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    max_b = len(val_loader) if args.max_batches is None else min(len(val_loader), int(args.max_batches))
    print(
        f"[infer] start  ckpt={ckpt_path.name}  epoch_in_ckpt={int(payload.get('epoch', -1))}  "
        f"batches≈{max_b}  viz_n={args.viz_n}  log_interval={args.log_interval}",
        flush=True,
    )

    epoch = int(payload.get("epoch", -1))
    val_loss_ckpt = float(payload.get("val_loss", float("nan")))
    train_loss_ckpt = float(payload.get("train_loss", float("nan")))

    sum_loss = 0.0
    n_batches = 0
    sum_recon = 0.0
    sum_phi = 0.0
    sum_rho = sum_vx = sum_vy = sum_var = 0.0
    sum_obs_t = sum_prior_t = 0.0
    viz_saved = 0

    # Solver 使用 autograd.grad；不能用外层 torch.no_grad()
    for bi, batch in enumerate(val_loader):
        if args.max_batches is not None and bi >= args.max_batches:
            break
        history, obs, obs_mask, x_gt = [b.to(device) for b in batch]
        loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
        sum_loss += float(loss.item())
        sum_recon += info["recon"]
        sum_phi += info["phi_mse"]
        sum_rho += info["rho_mse"]
        sum_vx += info["vx_mse"]
        sum_vy += info["vy_mse"]
        sum_var += info["var_mse"]
        sum_obs_t += info["obs_term"]
        sum_prior_t += info["prior_term"]
        n_batches += 1
        li = int(args.log_interval)
        if li > 0 and (bi % li == 0 or bi + 1 == max_b):
            mean_so_far = sum_loss / n_batches
            print(
                f"  batch {bi + 1}/{max_b}  loss={loss.item():.6f}  run_mean={mean_so_far:.6f}  "
                f"rho={info['rho_mse']:.4f}  vx={info['vx_mse']:.4f}  vy={info['vy_mse']:.4f}  "
                f"var={info['var_mse']:.4f}  |  obs_t={info['obs_term']:.4f}  prior_t={info['prior_term']:.4f}  "
                f"phi_mse={info['phi_mse']:.4f}",
                flush=True,
            )

        if args.viz_n > 0 and viz_saved < args.viz_n:
            x_hat = model.forward(history, obs, obs_mask, x_gt)
            B = x_gt.shape[0]
            for i in range(B):
                if viz_saved >= args.viz_n:
                    break
                gt = x_gt[i].detach().cpu().numpy()
                xh = x_hat[i].detach().cpu().numpy()
                ob = obs[i].detach().cpu().numpy()
                mk = obs_mask[i, 0].detach().cpu().numpy()
                partial = _partial_obs_as_nan(ob, mk)
                _save_step_plot_project_style(partial, gt, xh, viz_saved, out_dir)
                print(f"  saved {out_dir / f'step_{viz_saved:03d}.png'}", flush=True)
                viz_saved += 1

    nb = max(n_batches, 1)
    metrics = {
        "ckpt": str(ckpt_path),
        "epoch_saved": epoch,
        "val_loss_in_ckpt": val_loss_ckpt,
        "train_loss_in_ckpt": train_loss_ckpt,
        "val_batches_run": n_batches,
        "mean_val_loss": sum_loss / nb,
        "mean_recon_term": sum_recon / nb,
        "mean_phi_mse_prior_vs_gt": sum_phi / nb,
        "mean_rho_mse": sum_rho / nb,
        "mean_vx_mse": sum_vx / nb,
        "mean_vy_mse": sum_vy / nb,
        "mean_var_mse": sum_var / nb,
        "mean_obs_term": sum_obs_t / nb,
        "mean_prior_term": sum_prior_t / nb,
        "rho_mask_thr": float(model.cost_fn.rho_mask_thr),
        "viz_n_requested": int(args.viz_n),
        "viz_png_saved": int(viz_saved),
        "meta": meta,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        f"[infer] ckpt_epoch={epoch} ckpt_val_loss={val_loss_ckpt:.6f} "
        f"mean_loss({n_batches}b)={metrics['mean_val_loss']:.6f}  "
        f"mean_rho={metrics['mean_rho_mse']:.4f}  mean_vx={metrics['mean_vx_mse']:.4f}  "
        f"mean_vy={metrics['mean_vy_mse']:.4f}  mean_var={metrics['mean_var_mse']:.4f}  "
        f"viz_saved={viz_saved} png_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
