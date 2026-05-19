"""
训练 CrowdVarNet（**全程 rollout + TBPTT** 版本）。

核心差异（vs ``train_cli``）：
- 每个 sample 是长度 ``--episode-len`` 的连续 episode（默认 300 步），不是单个目标帧；
- 前 ``--warmup`` 步 history 用 GT 做热身，之后 history 每步压入自估 ``x_hat``；
- 每 ``--k-prime`` 步构成一个 TBPTT 窗口：窗口末 backward + step + detach，继续下一个窗口。

推理路径保持 ``CrowdVarNet.forward(history, obs, obs_mask)`` 不变；训练也只走该路径。
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
    p.add_argument("--checkpoint", type=str, required=True, help="PedPred .pth / .hkl")
    p.add_argument("--arch", type=str, default="pedpred3_gru_mid",
                   choices=("pedpred3_gru_mid",),
                   help="Only pedpred3_gru_mid is supported (final v13 teacher).")
    p.add_argument("--warmstart", type=str, default=None, help="既有 CrowdVarNet best.pt，继续训")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch", type=int, default=4, help="episode-batch（每个元素一整段 episode）")
    p.add_argument("--episode-len", type=int, default=300)
    p.add_argument("--k-prime", type=int, default=8, help="TBPTT 窗口长度")
    p.add_argument("--warmup", type=int, default=5, help="GT 热身步数（= T_hist）")
    p.add_argument("--grad-clip", type=float, default=0.5)
    p.add_argument("--sensing-range", type=float, default=5.0)
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument("--w-prior", type=float, default=0.5)
    p.add_argument("--rho-mask-thr", type=float, default=0.05)
    p.add_argument(
        "--solver-type",
        type=str,
        default="convgru",
        choices=("convgru",),
        help="Only convgru solver is supported (final winning combo).",
    )
    p.add_argument("--solver-hidden", type=int, default=256, help="convgru 隐藏通道数")
    p.add_argument("--solver-kernel", type=int, default=3, help="convgru 卷积核大小")
    p.add_argument(
        "--solver-no-share",
        action="store_true",
        help="convgru 默认每步共享权重；加此 flag 改为每步独立权重（参数 ×n_iter）",
    )
    p.add_argument("--train-workers", type=int, default=0)
    p.add_argument("--val-workers", type=int, default=0)
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--val-max-episodes", type=int, default=32, help="val 阶段最多跑这么多 episode")
    # Tier S 优化：EMA + AdamW + Dropout
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight_decay；>0 时使用 AdamW，否则 Adam")
    p.add_argument("--solver-dropout", type=float, default=0.0,
                   help="solver _OutHead 中 Dropout2d 概率（0=禁用）")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="权重 EMA β（如 0.9999）；0 表示不启用 EMA")
    p.add_argument("--lr-eta-min-ratio", type=float, default=0.2,
                   help="cosine 退火终点 lr 占初始 lr 的比例（默认 0.2）")
    p.add_argument("--flip-w-prob", type=float, default=0.0,
                   help="训练时以此概率沿 W 轴翻转 episode（vx 取反）。仅训练增强，val 不增强")
    p.add_argument("--obs-noise-std", type=float, nargs=4, default=None,
                   metavar=("RHO", "VX", "VY", "VAR"),
                   help="训练 obs 加高斯噪声的 σ（4 通道）。例：0.057 0.317 0.086 0.0064。仅训练增强")
    p.add_argument("--unfreeze-phi-tail", type=int, default=0,
                   help="解冻教师 forecaster 末端最后 N 层 Conv（默认 0=全冻结）。"
                        "PedPred_v4：N=1 解冻最终 1x1 conv；N=3 解冻末端 3 个 conv 块（~4.6k 参数）")
    p.add_argument("--phi-lr", type=float, default=None,
                   help="（仅当 --unfreeze-phi-tail>0 时有效）教师解冻部分用单独的 lr，"
                        "默认 = main_lr / 20（如 main_lr=2e-4 时教师 lr=1e-5）")
    p.add_argument("--ch-weights", type=float, nargs=4, default=None,
                   metavar=("RHO", "VX", "VY", "VAR"),
                   help="cost / recon loss 的 4 通道权重；默认 (2.5,1.5,1.0,0.5)。"
                        "示例：3.0 1.5 1.0 0.0 = 完全忽略 var 通道")
    if argv is None:
        argv = sys.argv[1:]
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    T_hist = int(args.warmup)

    base_train_loader, base_val_loader = get_atc_data(
        "train",
        "valid",
        batch=1,  # 重新打包，这里只借用底层 ConcatDataset 结构
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
        seed=17,
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
        seed=113,
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

    ped = load_frozen_pedpred(args.checkpoint, device, arch=args.arch)
    solver_type = args.solver_type  # only "convgru" is allowed by argparse
    ch_weights = tuple(args.ch_weights) if args.ch_weights else (2.5, 1.5, 1.0, 0.5)
    print(f"[rollout-train] ch_weights = {ch_weights}", flush=True)
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=T_hist,
        n_iter=args.n_iter,
        use_gru=False,
        w_prior=args.w_prior,
        ch_weights=ch_weights,
        rho_mask_thr=float(args.rho_mask_thr),
        solver_type=solver_type,
        solver_hidden=args.solver_hidden,
        solver_kernel=args.solver_kernel,
        solver_share=not args.solver_no_share,
        solver_dropout=float(args.solver_dropout),
        init_gate=False,
        unfreeze_phi_tail=int(args.unfreeze_phi_tail),
    ).to(device)

    if args.warmstart:
        ws = torch.load(args.warmstart, map_location=device)
        sd = ws["model_state_dict"] if isinstance(ws, dict) and "model_state_dict" in ws else ws
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            f"[rollout-train] warmstart from {args.warmstart}: "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    trainable = [p for p in model.parameters() if p.requires_grad]
    # 区分 solver/init_gate（用 main lr）vs 教师解冻部分（用 phi_lr）
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

    # EMA shadow weights（推理/eval 用 shadow）
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
        """临时把 EMA 副本换进 model.state_dict 以做 eval；返回原副本（dict 浅拷贝）。"""
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
            "w_prior": args.w_prior,
            "rho_mask_thr": args.rho_mask_thr,
            "solver_type": solver_type,
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
            "phi_lr": float(args.phi_lr) if args.phi_lr is not None else (args.lr / 20.0),
            "ch_weights": list(ch_weights),
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
            model,
            train_loader,
            opt,
            device,
            warmup=args.warmup,
            k_prime=args.k_prime,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            epoch=epoch,
            num_epochs=args.epochs,
            after_step_callback=_ema_update if ema_state is not None else None,
        )

        # val：最多跑 args.val_max_episodes 个 batch（节省墙钟）
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

        # 启用 EMA 时：val 与 best 评估都用 EMA 副本
        ema_backup = _swap_in_ema()
        try:
            va = rollout_val_loss(
                model,
                _ListLoader(capped_val_iter),
                device,
                warmup=args.warmup,
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
                # best.pt 中 model_state_dict 用 EMA 副本（因为 val 也是 EMA 评估的）
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
