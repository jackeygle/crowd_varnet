"""量化 ckpt 在密度通道上的「非支撑泄漏」。

跑 val（或 train），对每个样本算：
  rho_mse_global   全图 MSE
  rho_mse_support  GT 密度 > rho_thr 的格点 MSE（与训练 metrics 对齐）
  rho_mse_bg       GT 密度 <= rho_thr 的格点 MSE（背景）
  rho_mean_bg      背景上预测密度均值（绝对值，反映"蓝色铺底"幅度）
  rho_frac_bg_gt_thr 背景上预测密度 > rho_thr 的像素占比

用法::
    python -m scripts.diag_density_leak --ckpt /path/to/best.pt --device cuda --max-batches 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# 复用 infer_cli 的 wrap 逻辑
from crowd_varnet.cli import build_model_from_ckpt
from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.train_cli import wrap_loader_varnet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-batches", type=int, default=30)
    p.add_argument("--val-workers", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="可选：把汇总写入 JSON")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    model, meta = build_model_from_ckpt(ckpt_path, device=device)
    model.eval()
    rho_thr = float(model.cost_fn.rho_mask_thr)
    nin = int(model.T_hist)

    default_batch = int(os.environ.get("PEDPRED_BATCH", "16"))
    batch = int(meta.get("batch", default_batch))
    nout = int(meta.get("nout", os.environ.get("PEDPRED_NOUT", "1")))
    obs_mode = str(meta.get("obs_mode", "sensor"))
    sensing_range = float(meta.get("sensing_range", 5.0))
    num_agents = int(meta.get("num_agents", 3))

    val_loader = get_atc_data(
        "valid", batch=batch, nin=nin, nout=nout,
        num_workers=args.val_workers, drop_last=False,
    )
    val_loader = wrap_loader_varnet(
        val_loader, seed=1, obs_mode=obs_mode, partial_frac=0.35,
        sensing_range=sensing_range, num_agents=num_agents,
        batch_size=batch, num_workers=args.val_workers,
        shuffle=False, drop_last=False,
    )

    n_pix_total = 0
    n_pix_support = 0
    n_pix_bg = 0
    n_pix_bg_pred_pos = 0
    sum_se_total = 0.0
    sum_se_support = 0.0
    sum_se_bg = 0.0
    sum_abs_bg = 0.0
    n_batches_run = 0

    saved = [(p_, p_.requires_grad) for p_ in model.parameters() if p_.requires_grad]
    for p_, _ in saved:
        p_.requires_grad_(False)
    try:
        for bi, batch_data in enumerate(val_loader):
            if bi >= args.max_batches:
                break
            history, obs, obs_mask, x_gt = [b.to(device) for b in batch_data]
            x_hat = model.forward(history, obs, obs_mask).detach()
            rho_hat = x_hat[:, 0:1]
            rho_gt = x_gt[:, 0:1]
            err2 = (rho_hat - rho_gt).pow(2)
            support = (rho_gt > rho_thr).float()
            bg = 1.0 - support

            n_pix_total += rho_gt.numel()
            n_s = float(support.sum().item())
            n_b = float(bg.sum().item())
            n_pix_support += n_s
            n_pix_bg += n_b
            n_pix_bg_pred_pos += float(((rho_hat > rho_thr).float() * bg).sum().item())

            sum_se_total += float(err2.sum().item())
            sum_se_support += float((err2 * support).sum().item())
            sum_se_bg += float((err2 * bg).sum().item())
            sum_abs_bg += float((rho_hat.abs() * bg).sum().item())
            n_batches_run += 1
            if (bi + 1) % 5 == 0 or bi + 1 == args.max_batches:
                run_glob = sum_se_total / max(n_pix_total, 1)
                run_sup = sum_se_support / max(n_pix_support, 1)
                run_bg = sum_se_bg / max(n_pix_bg, 1)
                run_meanbg = sum_abs_bg / max(n_pix_bg, 1)
                run_frac = n_pix_bg_pred_pos / max(n_pix_bg, 1)
                print(
                    f"  batch {bi + 1}/{args.max_batches}  "
                    f"rho_mse_global={run_glob:.6f}  rho_mse_support={run_sup:.6f}  "
                    f"rho_mse_bg={run_bg:.6f}  rho_mean_bg={run_meanbg:.6f}  "
                    f"frac_bg_pred_gt_thr={run_frac:.4f}",
                    flush=True,
                )
    finally:
        for p_, was in saved:
            p_.requires_grad_(was)

    out = {
        "ckpt": str(ckpt_path),
        "rho_mask_thr": rho_thr,
        "n_batches_run": n_batches_run,
        "n_pix_total": int(n_pix_total),
        "n_pix_support": int(n_pix_support),
        "n_pix_bg": int(n_pix_bg),
        "rho_mse_global": sum_se_total / max(n_pix_total, 1),
        "rho_mse_support": sum_se_support / max(n_pix_support, 1),
        "rho_mse_bg": sum_se_bg / max(n_pix_bg, 1),
        "rho_mean_bg": sum_abs_bg / max(n_pix_bg, 1),
        "frac_bg_pred_gt_thr": n_pix_bg_pred_pos / max(n_pix_bg, 1),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
