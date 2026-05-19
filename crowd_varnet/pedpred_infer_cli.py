"""
PedPred（老师 / crowd field）开环一步推理与可视化。

在 ATC grid_cache 验证或测试集上：用历史 ``nin`` 帧预测下一帧，与真值比 MSE，并保存 PNG（预测 | 真值 | 密度误差）。

用法（仓库根目录）::

    export PYTHONPATH=/scratch/work/zhangx29/crowd_varnet
    export PEDPRED_RESOLUTION=1.0 PEDPRED_PERIOD=1.0 PEDPRED_KERNEL=tri
    export PEDPRED_BATCH=8 PEDPRED_NIN=5 PEDPRED_NOUT=1

    python -m crowd_varnet.pedpred_infer_cli \\
      --checkpoint /scratch/work/zhangx29/crowd_varnet/runs/pedpred_v4_final_17743578/checkpoints/amazed-finch_best.hkl \\
      --arch pedpred3 \\
      --split valid \\
      --out-dir /scratch/work/zhangx29/crowd_varnet/runs/pedpred_infer_vis \\
      --viz-n 8 \\
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

from .assimilation_model import FrozenPedPredPrior, load_frozen_pedpred
from .deps.dataset_atc import get_atc_data
from .deps.utils_plot import plot_generated_matrix_on_ax


def _as_bthcw(x: torch.Tensor) -> torch.Tensor:
    """确保 history 为 [B,T,4,H,W] float。"""
    if x.dim() == 5:
        return x.float()
    raise ValueError(f"expected history [B,T,4,H,W], got {tuple(x.shape)}")


def _as_b4hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5 and x.shape[1] == 1:
        return x[:, 0].float()
    if x.dim() == 4:
        return x.float()
    raise ValueError(f"expected target [B,1,4,H,W] or [B,4,H,W], got {tuple(x.shape)}")


def _save_pred_gt_fig(
    pred_hw4: np.ndarray,
    gt_hw4: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    rho_e = np.abs(pred_hw4[0].astype(np.float64) - gt_hw4[0].astype(np.float64))
    emean = float(np.mean(rho_e))
    elocmax = float(np.max(rho_e))

    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.2)
    ax0 = fig.add_subplot(gs[0])
    im0 = plot_generated_matrix_on_ax(pred_hw4, ax0)
    ax0.set_title(f"{title}\nPedPred pred")

    ax1 = fig.add_subplot(gs[1])
    im1 = plot_generated_matrix_on_ax(gt_hw4, ax1)
    ax1.set_title("Ground truth (next frame)")

    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(rho_e, cmap="Reds", origin="upper", aspect="auto", vmin=0.0)
    ax2.set_title(f"|ρ_pred−ρ_gt|\nmean={emean:.4f} max={elocmax:.4f}")
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PedPred one-step open-loop inference + PNG metrics")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="/scratch/work/zhangx29/crowd_varnet/runs/pedpred_v4_final_17743578/checkpoints/amazed-finch_best.hkl",
        help="Teacher .pth (key 'model' or full state dict)",
    )
    p.add_argument(
        "--arch",
        type=str,
        default="pedpred3",
        choices=("pedpred", "pedpred2", "pedpred3", "pedpred3_partial_observation"),
        help="pedpred3=crowd_varnet.deps.pedpred_models.PedPred_Optimized；"
        "pedpred3_partial_observation=旧 PA 仓库 PedPred3",
    )
    p.add_argument("--split", type=str, default="valid", choices=("valid", "test"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--batch", type=int, default=None, help="默认: PEDPRED_BATCH 或 8")
    p.add_argument("--nin", type=int, default=None, help="默认: PEDPRED_NIN 或 5")
    p.add_argument("--nout", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=None, help="限制 batch 数（默认跑满）")
    p.add_argument("--viz-n", type=int, default=8, help="保存前 N 个样本的 pedpred_sample_XXX.png")
    return p.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = int(args.batch if args.batch is not None else os.environ.get("PEDPRED_BATCH", "8"))
    nin = int(args.nin if args.nin is not None else os.environ.get("PEDPRED_NIN", "5"))

    phi = load_frozen_pedpred(args.checkpoint, device, arch=args.arch)
    adapter = FrozenPedPredPrior(phi, freeze=True).to(device)
    adapter.eval()

    loader = get_atc_data(
        args.split,
        batch=batch,
        nin=nin,
        nout=args.nout,
        num_workers=args.num_workers,
        drop_last=False,
    )

    sum_mse = 0.0
    sum_mae_ch = torch.zeros(4, device=device)
    n_pix = 0
    n_batches = 0
    viz_done = 0

    with torch.no_grad():
        for bi, batch_ in enumerate(loader):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            hist, tgt = batch_
            hist = _as_bthcw(hist.to(device))
            tgt = _as_b4hw(tgt.to(device))
            pred = adapter(hist)
            diff = pred - tgt
            sum_mse += float(diff.pow(2).mean().item())
            b, _, h, w = pred.shape
            sum_mae_ch += diff.abs().sum(dim=(0, 2, 3))
            n_pix += b * h * w
            n_batches += 1

            if args.viz_n > 0 and viz_done < args.viz_n:
                for i in range(b):
                    if viz_done >= args.viz_n:
                        break
                    ph = pred[i].detach().cpu().numpy()
                    gh = tgt[i].detach().cpu().numpy()
                    _save_pred_gt_fig(
                        ph,
                        gh,
                        out_dir / f"pedpred_sample_{viz_done:03d}.png",
                        title=f"split={args.split} batch={bi} idx={i}",
                    )
                    viz_done += 1

    mean_mse = sum_mse / max(n_batches, 1)
    mae_ch = (sum_mae_ch / max(n_pix, 1)).cpu().tolist()
    summary: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "arch": args.arch,
        "split": args.split,
        "nin": nin,
        "nout": args.nout,
        "batch": batch,
        "batches": n_batches,
        "mean_mse_all_channels": mean_mse,
        "mean_abs_err_per_channel_rho_vx_vy_var": mae_ch,
        "viz_png_saved": viz_done,
        "out_dir": str(out_dir),
    }
    (out_dir / "pedpred_infer_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[pedpred_infer] split={args.split} batches={n_batches} mean_MSE={mean_mse:.6f} "
        f"MAE_ch={mae_ch} png={viz_done} -> {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
