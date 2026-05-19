"""速度通道的"噪声地板"诊断：估计任何模型在 valid 上能达到的速度 RMSE 下限。

两个互补的基线：

1. **persistence**（持续基线）：把输入最后一帧的速度直接当下一帧的预测。
   - 如果 teacher 比 persistence 好 30%+，说明它在学真实动力学。
   - 如果只好 5%，模型几乎没学到什么。

2. **temporal smoothing 对比**：把 GT 速度做一次帧间均值（smooth_v = (v_T-1 + v_T + v_T+1)/3），
   再和原始 v_T+1 比较。这个差异就是 GT 速度本身的高频噪声 —— 没有时序平滑的预测器永远跨不过去。

输出：和 eval_teacher_mse 同一掩码（rho > 0.05）下的 vx/vy support_RMSE 数字，可直接横向比较。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.deps.grid_data import GridData


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-batches", type=int, default=100)
    ap.add_argument("--rho-thr", type=float, default=0.05)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    if args.data_dir:
        os.environ["PEDPRED_ATC_DATA_DIR"] = args.data_dir

    print(f"[noise floor] data_dir={os.environ.get('PEDPRED_ATC_DATA_DIR','default')}")
    print(f"[noise floor] rho_thr={args.rho_thr}  max_batches={args.max_batches}")

    # 用 nin=5, nout=1：input [B,5,4,H,W]，target [B,1,4,H,W]
    val_loader = get_atc_data("valid", batch=8, nin=5, nout=1,
                              num_workers=0, drop_last=False, pin_memory=False)

    # === 基线 1: persistence ===
    sum_se_pers = np.zeros(2)        # vx, vy
    # === 基线 2: temporal smoothing 残差（GT 自己 vs 邻帧均值） ===
    sum_se_smooth = np.zeros(2)
    n_pix_support = 0

    # 额外统计：GT 速度本身的方差（衡量信号尺度）
    sum_v_sq = np.zeros(2)
    sum_v = np.zeros(2)
    n_pix_for_var = 0

    for bi, (inp, tgt) in enumerate(val_loader):
        if bi >= args.max_batches:
            break

        inp_t = GridData(inp).as_tensor("density", "vel_mean", "vel_var")
        tgt_t = GridData(tgt).as_tensor("density", "vel_mean", "vel_var")

        # 形状: inp_t [B,5,4,H,W], tgt_t [B,1,4,H,W]
        # rho 通道 = 0, vx = 1, vy = 2
        last_inp_v   = inp_t[:, -1, 1:3].numpy()           # [B,2,H,W] (T=5 时刻速度)
        prev_inp_v   = inp_t[:, -2, 1:3].numpy()           # [B,2,H,W] (T=4)
        target_v     = tgt_t[:, 0, 1:3].numpy()            # [B,2,H,W] (T=6 GT)
        target_rho   = tgt_t[:, 0, 0].numpy()              # [B,H,W]

        # support mask: GT 密度 > 阈值
        support = (target_rho > args.rho_thr).astype(np.float32)  # [B,H,W]
        support_v = support[:, None, :, :]                         # [B,1,H,W]

        # persistence: pred = last_inp_v
        err_pers = (last_inp_v - target_v) ** 2  # [B,2,H,W]
        for c in range(2):
            sum_se_pers[c] += (err_pers[:, c] * support).sum()

        # temporal smoothing baseline: smooth = (prev + last + target) / 3，残差 = target - smooth
        smooth = (prev_inp_v + last_inp_v + target_v) / 3.0
        err_smooth = (target_v - smooth) ** 2  # [B,2,H,W]
        for c in range(2):
            sum_se_smooth[c] += (err_smooth[:, c] * support).sum()

        # GT signal 方差
        for c in range(2):
            sum_v_sq[c] += (target_v[:, c] ** 2 * support).sum()
            sum_v[c]    += (target_v[:, c]      * support).sum()

        n_support_b = int(support.sum())
        n_pix_support  += n_support_b
        n_pix_for_var  += n_support_b

        if (bi + 1) % 20 == 0:
            mse = sum_se_pers[0] / max(n_pix_support, 1)
            print(f"  batch {bi+1}/{args.max_batches}  persistence vx_MSE={mse:.5f}", flush=True)

    n_safe = max(n_pix_support, 1)
    print()
    print(f"=== Noise-floor results (over {bi+1} batches, {n_pix_support} support pixels) ===")
    print(f"{'estimator':>22} | {'vx_MSE':>10} | {'vx_RMSE':>10} | {'vy_MSE':>10} | {'vy_RMSE':>10}")

    for name, sum_se in [
        ("persistence (v_T → v_T+1)", sum_se_pers),
        ("smoothing residual    ", sum_se_smooth),
    ]:
        mx, my = sum_se[0] / n_safe, sum_se[1] / n_safe
        print(f"{name:>22} | {mx:>10.5f} | {np.sqrt(mx):>10.5f} | {my:>10.5f} | {np.sqrt(my):>10.5f}")

    print()
    print("=== GT signal scale (support pixels only) ===")
    n_safe_v = max(n_pix_for_var, 1)
    for c, name in [(0, "vx"), (1, "vy")]:
        mean = sum_v[c] / n_safe_v
        var  = sum_v_sq[c] / n_safe_v - mean ** 2
        print(f"  {name}: mean={mean:+.4f}  std={np.sqrt(max(var,0)):.4f}")

    print()
    print("解释：")
    print("  - persistence RMSE 是 'do-nothing' 模型的成绩；teacher 应该比这个好。")
    print("  - smoothing residual RMSE 是 GT 高频噪声的估计；任何时间平滑预测器都跨不过这个下限。")
    print("  - GT std 给出信号尺度：噪声 / 信号 比可以判断信噪比。")


if __name__ == "__main__":
    main()
