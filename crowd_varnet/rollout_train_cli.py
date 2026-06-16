"""
训练 CrowdVarNet（全程 rollout + TBPTT）。

每个 sample 是长度 --episode-len 的连续 episode（默认 300 步）；
前 --warmup 步 history 用 GT 做热身，之后 history 每步压入自估 x_hat；
每 --k-prime 步构成一个 TBPTT 窗口：窗口末 backward + step + detach。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader

from .deps.dataset_atc import get_atc_data
from .datasets import RolloutEpisodeDataset, unwrap_concat_base_dataset
from .models import CrowdVarNet, load_frozen_pedpred
from .training import rollout_tbptt_epoch, rollout_val_loss


def _build_rollout_loader(
    loader: DataLoader,
    *,
    episode_len: int,
    sensing_range: float,
    num_agents: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
    flip_w_prob: float = 0.0,
    obs_noise_std: Optional[List[float]] = None,
) -> DataLoader:
    bases = list(unwrap_concat_base_dataset(loader.dataset))
    wrapped = [
        RolloutEpisodeDataset(
            b,
            episode_len=episode_len,
            sensing_range=sensing_range,
            num_agents=num_agents,
            seed=seed + i * 7919,
            flip_w_prob=flip_w_prob,
            obs_noise_std=obs_noise_std,
        )
        for i, b in enumerate(bases)
    ]
    ds = wrapped[0] if len(wrapped) == 1 else ConcatDataset(wrapped)

    dl_kw: dict = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=drop_last,
    )
    if num_workers > 0:
        dl_kw["prefetch_factor"] = 2
        dl_kw["persistent_workers"] = True
    return DataLoader(**dl_kw)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CrowdVarNet via full-rollout TBPTT")
    p.add_argument("--checkpoint", type=str, required=True, help="PedPred teacher .pth / .hkl")
    p.add_argument("--arch", type=str, default="pedpred3_gru_mid")
    p.add_argument("--use-their-teacher", action="store_true",
                   help="如果设置，--checkpoint 是他们的 PedPred3 ckpt（apt-ibex_*.pth），"
                        "通过 TheirPedPredAdapter 包一层（CPU 推理，nin=1）。"
                        "否则用我们的 load_frozen_pedpred 加载（默认）。")
    p.add_argument("--warmstart", type=str, default=None, help="既有 CrowdVarNet best.pt，继续训")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--episode-len", type=int, default=300)
    p.add_argument("--k-prime", type=int, default=8, help="TBPTT 窗口长度")
    p.add_argument("--warmup", type=int, default=5, help="GT 热身步数（= T_hist）")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--sensing-range", type=float, default=5.0)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument("--w-obs", type=float, default=1.0,
                   help="Total weight on observation term. Default 1.0.")
    p.add_argument("--w-prior", type=float, default=0.5)
    p.add_argument("--rho-mask-thr", type=float, default=0.05)
    p.add_argument("--prior-unobs-weight", type=float, default=0.05,
                   help="Weight on prior cost in unobserved regions (0=free, 1=same as observed). "
                        "Default 0.05 lets solver use attention to infer unobs regions; "
                        "values >1 force the unobserved cells to copy the teacher prior more strongly.")
    # ConvGRU solver 参数
    p.add_argument("--solver-hidden", type=int, default=256, help="ConvGRU 隐藏通道数")
    p.add_argument("--solver-kernel", type=int, default=3, help="ConvGRU 卷积核大小")
    p.add_argument("--solver-no-share", action="store_true",
                   help="每步独立权重（默认共享）")
    p.add_argument("--solver-dropout", type=float, default=0.0)
    p.add_argument("--solver-no-attention", action="store_true",
                   help="禁用 spatial attention（默认启用）")
    p.add_argument("--solver-attn-heads", type=int, default=4,
                   help="Spatial attention head 数")
    p.add_argument("--solver-momentum", type=float, default=0.5,
                   help="Momentum 系数初始值（0=关闭，0.5=默认，仅 gru 模式有效）")
    p.add_argument("--solver-rnn-type", type=str, default="gru",
                   choices=("gru", "lstm"),
                   help="Solver RNN cell 类型：gru=ConvGRU+显式momentum; "
                        "lstm=ConvLSTM(cell state 隐式动量，跟 4DVarNet 对齐)")
    p.add_argument("--solver-lr-grad", type=float, default=0.0,
                   help="4DVarNet-style direct gradient weight (0=disabled, 0.2=4DVarNet default). "
                        "When >0, update = delta + lr_grad * (step+1)/n_iter * grad. "
                        "This pushes solver to follow cost gradient even in unobs regions.")
    p.add_argument("--solver-use-obs-encoder", action="store_true",
                   help="Enable global observation encoder (Perceiver-style): "
                        "Transformer encodes all obs pixels into tokens; solver hidden "
                        "queries them via cross-attention. Lets unobs regions directly "
                        "attend to all observations, not just nearby pixels.")
    # 数据与训练
    p.add_argument("--train-workers", type=int, default=0)
    p.add_argument("--val-workers", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--val-max-episodes", type=int, default=32)
    # 优化：EMA + AdamW + Dropout
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--ema-decay", type=float, default=0.0)
    p.add_argument("--lr-eta-min-ratio", type=float, default=0.2)
    # 数据增强
    p.add_argument("--flip-w-prob", type=float, default=0.0)
    p.add_argument("--obs-noise-std", type=float, nargs=4, default=None,
                   metavar=("RHO", "VX", "VY", "VAR"))
    # 教师微调（可选）
    p.add_argument("--unfreeze-phi-tail", type=int, default=0)
    p.add_argument("--phi-lr", type=float, default=None)
    # 通道权重
    p.add_argument("--ch-weights", type=float, nargs=4, default=None,
                   metavar=("RHO", "VX", "VY", "VAR"))
    # === Loss 优化（新加的，默认关闭以保持向后兼容）===
    p.add_argument("--lambda-bg", type=float, default=0.0,
                   help="优化 1: 背景密度抑制权重（建议 0.5-2.0；0=关闭）")
    p.add_argument("--lookahead-gamma", type=float, default=1.0,
                   help="优化 2: TBPTT 窗口内 lookahead 增长率（gamma**t；1.0=关闭，1.1-1.3=远期步加权）")
    p.add_argument("--lambda-dir", type=float, default=0.0,
                   help="优化 4: 速度方向 cosine 损失权重（建议 0.1-0.5；0=关闭）")
    p.add_argument("--lambda-mass", type=float, default=0.0,
                   help="优化 5: 质量守恒损失权重（|sum(rho)-sum(rho_gt)|/sum；建议 0.1-0.5；0=关闭）")
    p.add_argument("--lambda-log-rho", type=float, default=0.0,
                   help="优化 6: log-space 密度 MSE 权重（在小值区域更敏感；建议 0.3-1.0；0=关闭）")
    p.add_argument("--weight-mode", type=str, default="hard_mask",
                   choices=("hard_mask", "continuous", "full"),
                   help="速度通道 loss 加权方式："
                        "hard_mask=二值掩码 (rho>thr); "
                        "continuous=他们风格 (target_rho 连续加权); "
                        "full=全图所有通道都算 loss（鼓励未观测区也准）")
    p.add_argument("--unobs-loss-weight", type=float, default=1.0,
                   help="未观测区有人处的 loss 权重倍数 (1.0=不加权, 5.0=强调5倍). "
                        "防止 loss 被背景稀释，逼模型学习推断未观测区。")
    p.add_argument("--sched-sampling-prob", type=float, default=0.0,
                   help="Scheduled sampling: probability of pushing GT instead of self-estimate "
                        "into history buffer during training. 0=pure rollout, 0.3=30%% GT anchors. "
                        "Mitigates rollout drift; inference still uses pure self-estimates.")
    p.add_argument("--lambda-vel-sparsity", type=float, default=0.0,
                   help="Velocity sparsity regularizer. Penalizes velocity magnitude in GT "
                        "background regions (density < thr). 0.5-2.0 recommended to reduce "
                        "velocity over-extension in unobserved regions.")
    p.add_argument("--lambda-density", type=float, default=0.0,
                   help="Density-only auxiliary MSE loss (full image). "
                        "1.0-3.0 recommended to strengthen density learning.")
    p.add_argument("--topk-percent", type=float, default=1.0,
                   help="OHEM-style top-k% hard pixel mining: only the top fraction of "
                        "largest-error pixels contribute to recon loss. 1.0=disabled, "
                        "0.2=top 20%% hardest. Inspired by Motion-DeepLab top-k cross entropy.")
    p.add_argument("--loss-style", type=str, default="mse",
                   choices=("mse", "teacher", "detached_nll"),
                   help="主重建 loss 形式："
                        "mse=baseline (通道权重 + mask + MSE 或 NLLL); "
                        "teacher=PedPred3 风格 (het-NLLL on all channels, ρ_gt-weighted vel); "
                        "detached_nll=MSE 驱动 solver + detached NLLL 驱动 var_head（推荐）。"
                        "detached_nll 模式下 solver 精度 = baseline MSE，var_head 独立学 σ。")
    p.add_argument("--teacher-lambda-vel", type=float, default=1.0,
                   help="loss-style=teacher 时速度 NLL 的权重（默认 1.0，跟教师一致）")
    p.add_argument("--teacher-lambda-rho", type=float, default=1.0,
                   help="loss-style=teacher 时密度 NLL 的权重（默认 1.0；建议 3.0 强调密度）")
    p.add_argument("--teacher-lambda-vx", type=float, default=1.0,
                   help="loss-style=teacher 时 vx NLL 的权重（默认 1.0；建议 1.5）")
    p.add_argument("--teacher-lambda-vy", type=float, default=1.0,
                   help="loss-style=teacher 时 vy NLL 的权重（默认 1.0）")
    # === Uncertainty quantification (Stage B/C) ===
    p.add_argument("--predict-uncertainty", action="store_true",
                   help="启用 var_head 输出 per-pixel log_var 并改用异方差 NLLL；"
                        "用于 Deep Ensemble 时分解 aleatoric / epistemic uncertainty")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子（控制 init / dataloader / dropout）。"
                        "Deep Ensemble 用不同 --seed 训 N 个独立模型。")
    if argv is None:
        argv = sys.argv[1:]
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    T_hist = int(args.warmup)

    # Seed everything for reproducibility & ensemble member independence
    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)
    print(f"[rollout-train] seed = {seed}", flush=True)

    base_train_loader, base_val_loader = get_atc_data(
        "train", "valid",
        batch=1,
        nin=T_hist,
        nout=1,
        num_workers=0,
        validation_num_workers=0,
    )

    train_loader = _build_rollout_loader(
        base_train_loader,
        episode_len=args.episode_len,
        sensing_range=args.sensing_range,
        num_agents=args.num_agents,
        seed=17 + seed,
        batch_size=args.batch,
        num_workers=args.train_workers,
        shuffle=True,
        drop_last=True,
        flip_w_prob=float(args.flip_w_prob),
        obs_noise_std=args.obs_noise_std,
    )
    val_loader = _build_rollout_loader(
        base_val_loader,
        episode_len=args.episode_len,
        sensing_range=args.sensing_range,
        num_agents=args.num_agents,
        seed=113 + seed,
        batch_size=max(1, args.batch // 2),
        num_workers=args.val_workers,
        shuffle=False,
        drop_last=False,
        flip_w_prob=0.0,
        obs_noise_std=None,
    )

    print(
        f"[rollout-train] train_episodes={len(train_loader.dataset)} "
        f"val_episodes={len(val_loader.dataset)} "
        f"episode_len={args.episode_len} K'={args.k_prime} warmup={args.warmup}",
        flush=True,
    )

    if args.use_their_teacher:
        from .models.their_teacher_adapter import TheirPedPredAdapter
        ped = TheirPedPredAdapter(args.checkpoint, device=str(device))
        print(f"[rollout-train] using THEIR teacher (PedPred3 from Partial_observation): "
              f"{args.checkpoint}", flush=True)
    else:
        ped = load_frozen_pedpred(args.checkpoint, device, arch=args.arch)
    ch_weights = tuple(args.ch_weights) if args.ch_weights else (1.0, 1.0, 1.0, 0.0)
    print(f"[rollout-train] ch_weights = {ch_weights}", flush=True)

    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=T_hist,
        n_iter=args.n_iter,
        w_obs=args.w_obs,
        w_prior=args.w_prior,
        ch_weights=ch_weights,
        rho_mask_thr=float(args.rho_mask_thr),
        solver_hidden=args.solver_hidden,
        solver_kernel=args.solver_kernel,
        solver_share=not args.solver_no_share,
        solver_dropout=float(args.solver_dropout),
        unfreeze_phi_tail=int(args.unfreeze_phi_tail),
        predict_uncertainty=bool(args.predict_uncertainty),
        solver_use_attention=not args.solver_no_attention,
        solver_attn_heads=int(args.solver_attn_heads),
        solver_momentum=float(args.solver_momentum),
        solver_rnn_type=str(args.solver_rnn_type),
        prior_unobs_weight=float(args.prior_unobs_weight),
        solver_lr_grad=float(args.solver_lr_grad),
        solver_use_obs_encoder=bool(args.solver_use_obs_encoder),
    ).to(device)

    if args.warmstart:
        ws = torch.load(args.warmstart, map_location=device, weights_only=False)
        sd = ws["model_state_dict"] if isinstance(ws, dict) and "model_state_dict" in ws else ws
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[rollout-train] warmstart from {args.warmstart}: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    # 区分 solver（main lr）vs 教师解冻部分（phi_lr）
    trainable = [p for p in model.parameters() if p.requires_grad]
    phi_trainable = [p for p in model.adapter.phi.parameters() if p.requires_grad]
    phi_ids = {id(p) for p in phi_trainable}
    main_trainable = [p for p in trainable if id(p) not in phi_ids]
    phi_lr = float(args.phi_lr) if args.phi_lr is not None else (args.lr / 20.0)

    if phi_trainable:
        param_groups = [
            {"params": main_trainable, "lr": args.lr, "name": "main"},
            {"params": phi_trainable, "lr": phi_lr, "name": "phi_tail"},
        ]
    else:
        param_groups = [{"params": main_trainable, "lr": args.lr, "name": "main"}]

    if args.weight_decay > 0:
        opt = AdamW(param_groups, lr=args.lr, weight_decay=float(args.weight_decay))
        opt_name = "AdamW"
    else:
        opt = Adam(param_groups, lr=args.lr)
        opt_name = "Adam"

    eta_min = args.lr * float(args.lr_eta_min_ratio)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=eta_min)
    print(
        f"[rollout-train] optimizer={opt_name} main_lr={args.lr:.2e} wd={args.weight_decay:.2e} "
        f"cosine eta_min={eta_min:.2e}  solver_dropout={args.solver_dropout:.2f}  "
        f"ema_decay={args.ema_decay:.4f}  unfreeze_phi_tail={args.unfreeze_phi_tail} "
        f"phi_lr={phi_lr:.2e}",
        flush=True,
    )

    # EMA shadow weights
    ema_decay = float(args.ema_decay)
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    if ema_decay > 0:
        ema_state = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if v.dtype in (torch.float32, torch.float16, torch.bfloat16)
        }

    def _ema_update():
        if ema_state is None:
            return
        sd = model.state_dict()
        for k in ema_state:
            ema_state[k].mul_(ema_decay).add_(sd[k].detach(), alpha=1.0 - ema_decay)

    def _swap_in_ema():
        if ema_state is None:
            return None
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in ema_state}
        model.load_state_dict({**model.state_dict(), **ema_state}, strict=False)
        return backup

    def _swap_out_ema(backup):
        if backup is None:
            return
        model.load_state_dict({**model.state_dict(), **backup}, strict=False)

    save_dir: Optional[Path] = Path(args.save_dir).resolve() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        meta: Dict[str, Any] = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pedpred_ckpt": str(Path(args.checkpoint).resolve()),
            "arch": args.arch,
            "warmstart": str(Path(args.warmstart).resolve()) if args.warmstart else None,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch": args.batch,
            "episode_len": args.episode_len,
            "k_prime": args.k_prime,
            "warmup": args.warmup,
            "grad_clip": args.grad_clip,
            "nin": T_hist,
            "sensing_range": args.sensing_range,
            "num_agents": args.num_agents,
            "n_iter": args.n_iter,
            "w_obs": args.w_obs,
            "w_prior": args.w_prior,
            "rho_mask_thr": args.rho_mask_thr,
            "solver_hidden": int(args.solver_hidden),
            "solver_kernel": int(args.solver_kernel),
            "solver_share": not args.solver_no_share,
            "solver_dropout": float(args.solver_dropout),
            "weight_decay": float(args.weight_decay),
            "ema_decay": float(args.ema_decay),
            "lr_eta_min_ratio": float(args.lr_eta_min_ratio),
            "flip_w_prob": float(args.flip_w_prob),
            "obs_noise_std": list(args.obs_noise_std) if args.obs_noise_std else None,
            "unfreeze_phi_tail": int(args.unfreeze_phi_tail),
            "phi_lr": phi_lr,
            "ch_weights": list(ch_weights),
            "lambda_bg": float(args.lambda_bg),
            "lookahead_gamma": float(args.lookahead_gamma),
            "lambda_dir": float(args.lambda_dir),
            "lambda_mass": float(args.lambda_mass),
            "lambda_log_rho": float(args.lambda_log_rho),
            "weight_mode": str(args.weight_mode),
            "loss_style": str(args.loss_style),
            "teacher_lambda_vel": float(args.teacher_lambda_vel),
            "teacher_lambda_rho": float(args.teacher_lambda_rho),
            "teacher_lambda_vx": float(args.teacher_lambda_vx),
            "teacher_lambda_vy": float(args.teacher_lambda_vy),
            "predict_uncertainty": bool(args.predict_uncertainty),
            "seed": int(args.seed),
            "training_mode": "rollout_tbptt",
        }
        (save_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _fmt_components(comp: Dict[str, float], tag: str) -> str:
        bg_part = ""
        if "rho_bg_mse" in comp:
            bg_part = (
                f" | rho_bg_mse={comp['rho_bg_mse']:.4f} "
                f"rho_bg_mean={comp['rho_bg_mean']:.4f}"
            )
        return (
            f"  {tag}: rho={comp['rho']:.4f} vx={comp['vx']:.4f} "
            f"vy={comp['vy']:.4f} sp={comp['speed']:.4f} | "
            f"w*: rho={comp['rho_w']:.4f} vx={comp['vx_w']:.4f} "
            f"vy={comp['vy_w']:.4f} sp={comp['speed_w']:.4f}"
            f"{bg_part}"
        )

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        tr = rollout_tbptt_epoch(
            model, train_loader, opt, device,
            warmup=args.warmup,
            k_prime=args.k_prime,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            num_epochs=args.epochs,
            after_step_callback=_ema_update if ema_state is not None else None,
            lambda_bg=float(args.lambda_bg),
            lookahead_gamma=float(args.lookahead_gamma),
            lambda_dir=float(args.lambda_dir),
            lambda_mass=float(args.lambda_mass),
            lambda_log_rho=float(args.lambda_log_rho),
            weight_mode=str(args.weight_mode),
            loss_style=str(args.loss_style),
            teacher_lambda_vel=float(args.teacher_lambda_vel),
            teacher_lambda_rho=float(args.teacher_lambda_rho),
            teacher_lambda_vx=float(args.teacher_lambda_vx),
            teacher_lambda_vy=float(args.teacher_lambda_vy),
            unobs_loss_weight=float(args.unobs_loss_weight),
            sched_sampling_prob=float(args.sched_sampling_prob),
            lambda_vel_sparsity=float(args.lambda_vel_sparsity),
            lambda_density=float(args.lambda_density),
            topk_percent=float(args.topk_percent),
        )

        # val：最多跑 val_max_episodes 个 batch
        capped_val_iter = []
        for i, b in enumerate(val_loader):
            if i * max(1, args.batch // 2) >= args.val_max_episodes:
                break
            capped_val_iter.append(b)

        class _ListLoader:
            def __init__(self, items):
                self.items = items
            def __iter__(self):
                return iter(self.items)
            def __len__(self):
                return len(self.items)

        ema_backup = _swap_in_ema()
        try:
            va = rollout_val_loss(
                model, _ListLoader(capped_val_iter), device,
                warmup=args.warmup,
                lambda_bg=float(args.lambda_bg),
                lambda_dir=float(args.lambda_dir),
                lambda_mass=float(args.lambda_mass),
                lambda_log_rho=float(args.lambda_log_rho),
                weight_mode=str(args.weight_mode),
                loss_style=str(args.loss_style),
                teacher_lambda_vel=float(args.teacher_lambda_vel),
                teacher_lambda_rho=float(args.teacher_lambda_rho),
                teacher_lambda_vx=float(args.teacher_lambda_vx),
                teacher_lambda_vy=float(args.teacher_lambda_vy),
            )
        finally:
            _swap_out_ema(ema_backup)

        sched.step()
        lr_now = sched.get_last_lr()[0]
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={tr['total']:.6f}  val_rollout_loss={va['total']:.6f}  "
            f"lr={lr_now:.2e}",
            flush=True,
        )
        print(_fmt_components(tr, "train"), flush=True)
        print(_fmt_components(va, "val  "), flush=True)

        history.append({"epoch": epoch + 1, "lr": lr_now, "train": tr, "val": va})
        if save_dir is not None:
            (save_dir / "metrics.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": tr["total"],
                "val_loss": va["total"],
                "train_components": tr,
                "val_components": va,
            }
            if ema_state is not None:
                payload["ema_state_dict"] = {k: v.detach().cpu() for k, v in ema_state.items()}
            torch.save(payload, save_dir / "last.pt")
            if va["total"] < best_val:
                best_val = va["total"]
                if ema_state is not None:
                    best_payload = dict(payload)
                    best_payload["model_state_dict"] = {k: v.detach().cpu() for k, v in ema_state.items()}
                    best_payload["ema_used"] = True
                    torch.save(best_payload, save_dir / "best.pt")
                else:
                    torch.save(payload, save_dir / "best.pt")
                print(f"  saved best.pt (val_rollout_loss={best_val:.6f}) -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
