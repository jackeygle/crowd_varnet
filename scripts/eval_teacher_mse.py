"""快速评估教师 ckpt 的纯 MSE（vs NLLL）。

目的：NLLL 受变量分布敏感（baseline = 0.5 log 2πσ²），跨数据集不直接可比。
MSE 直接看预测/GT 残差，跨数据集比较更稳。

用法::
    python -m scripts.eval_teacher_mse --ckpt /path/to/teacher_best.hkl --arch pedpred3 \\
        --max-batches 50
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from crowd_varnet.models.prior import load_frozen_pedpred
from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.deps.grid_data import GridData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="pedpred3")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--rho-thr", type=float, default=0.05)
    ap.add_argument("--data-dir", default=None,
                    help="覆盖 PEDPRED_ATC_DATA_DIR")
    args = ap.parse_args()

    if args.data_dir:
        os.environ["PEDPRED_ATC_DATA_DIR"] = args.data_dir
    device = torch.device(args.device)

    teacher = load_frozen_pedpred(args.ckpt, device, arch=args.arch)
    teacher.eval()

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] arch={args.arch}  data_dir={os.environ.get('PEDPRED_ATC_DATA_DIR','default')}")

    # 用 nin=5, nout=1（教师是 nin=5, nout=5 训的，但只算 1 步预测的 MSE）
    val_loader = get_atc_data("valid", batch=8, nin=5, nout=1,
                              num_workers=0, drop_last=False, pin_memory=False)

    # 累计统计
    sum_se_global = np.zeros(4)
    sum_se_support = np.zeros(4)
    n_pix_global = 0
    n_pix_support = 0

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= args.max_batches:
                break
            inp, tgt = batch
            inp = inp.to(device)
            tgt = tgt.to(device)
            inp_t = GridData(inp).as_tensor("density", "vel_mean", "vel_var").to(device)
            tgt_t = GridData(tgt).as_tensor("density", "vel_mean", "vel_var").to(device)
            # Predict horizon=1
            out = teacher(inp_t, horizon=1)
            pred_t = GridData(out).as_tensor("density", "vel_mean", "vel_var").to(device)
            err2 = (pred_t - tgt_t).pow(2).detach().cpu().numpy()  # [B,1,4,H,W]
            err2 = err2[:, 0]  # [B, 4, H, W]
            tgt_np = tgt_t[:, 0].detach().cpu().numpy()  # [B, 4, H, W]
            support = (tgt_np[:, 0:1] > args.rho_thr).astype(np.float32)  # [B, 1, H, W]

            for c in range(4):
                sum_se_global[c] += err2[:, c].sum()
                sum_se_support[c] += (err2[:, c] * support[:, 0]).sum()

            B, _, H, W = err2.shape
            n_pix_global += B * H * W
            n_pix_support += int(support.sum())

            if (bi + 1) % 10 == 0:
                print(f"  batch {bi+1}/{args.max_batches}  "
                      f"running rho_mse_global={sum_se_global[0]/max(n_pix_global,1):.5f}  "
                      f"rho_mse_support={sum_se_support[0]/max(n_pix_support,1):.5f}",
                      flush=True)

    n_safe_g = max(n_pix_global, 1)
    n_safe_s = max(n_pix_support, 1)
    print()
    print(f"=== Results (over {bi+1} batches, ~{n_pix_global} pixels, {n_pix_support} support pixels) ===")
    names = ["rho", "vx", "vy", "var"]
    print(f"{'channel':>10} | {'global_MSE':>12} | {'support_MSE':>12} | {'support_RMSE':>13}")
    for c, name in enumerate(names):
        g = sum_se_global[c] / n_safe_g
        s = sum_se_support[c] / n_safe_s
        print(f"{name:>10} | {g:>12.6f} | {s:>12.6f} | {np.sqrt(s):>13.6f}")
    total_g = sum_se_global.sum() / (4 * n_safe_g)
    total_s = sum_se_support.sum() / (4 * n_safe_s)
    print(f"{'all':>10} | {total_g:>12.6f} | {total_s:>12.6f}")


if __name__ == "__main__":
    main()
