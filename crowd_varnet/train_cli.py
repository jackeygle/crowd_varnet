"""
训练 CrowdVarNet：ATC 数据经包内 ``deps.dataset_atc`` 读取（grid_cache H5）。

与 Partial_observation / EnKF **只对齐实验配置**（数据划分、ATC 离散化参数、部分观测几何与 agent 运动叙事），
**不对齐** EnKF 同化算法本身；后者属于 teacher baseline，不是本模型的方法。

示例::

    cd /scratch/work/zhangx29/crowd_varnet
    export PYTHONPATH=/scratch/work/zhangx29/crowd_varnet
    export PEDPRED_BATCH=16 PEDPRED_NIN=5 PEDPRED_NOUT=1 \\
           PEDPRED_RESOLUTION=1.0 PEDPRED_PERIOD=1.0 PEDPRED_KERNEL=tri

    python -m crowd_varnet.train_cli \\
        --checkpoint /path/to/pedpred.pth --arch pedpred3 --epochs 2 --device cuda
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
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

from .deps.dataset_atc import get_atc_data

from .core import (  # noqa: E402
    CrowdVarNet,
    CrowdVarNetDataset,
    load_frozen_pedpred,
    train_one_epoch,
)


def wrap_loader_varnet(
    loader: DataLoader,
    seed: int,
    *,
    obs_mode: str = "sensor",
    partial_frac: float = 0.35,
    sensing_range: float = 5.0,
    num_agents: int = 3,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    base_ds = loader.dataset
    ds_kw = dict(
        obs_mode=obs_mode,
        partial_frac=partial_frac,
        sensing_range=sensing_range,
        num_agents=num_agents,
    )
    if isinstance(base_ds, ConcatDataset):
        wrapped = ConcatDataset(
            [CrowdVarNetDataset(ds, seed=seed + i, **ds_kw) for i, ds in enumerate(base_ds.datasets)]
        )
    else:
        wrapped = CrowdVarNetDataset(base_ds, seed=seed, **ds_kw)

    nw = num_workers if num_workers is not None else loader.num_workers
    dl_kw: dict = dict(
        dataset=wrapped,
        batch_size=batch_size if batch_size is not None else loader.batch_size,
        shuffle=shuffle if shuffle is not None else getattr(loader, "shuffle", True),
        num_workers=nw,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=drop_last if drop_last is not None else getattr(loader, "drop_last", False),
        generator=getattr(loader, "generator", None),
    )
    if nw > 0:
        dl_kw["prefetch_factor"] = getattr(loader, "prefetch_factor", None) or 2
        dl_kw["persistent_workers"] = getattr(loader, "persistent_workers", False)
    return DataLoader(**dl_kw)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CrowdVarNet with frozen PedPred prior")
    p.add_argument("--checkpoint", type=str, required=True, help="PedPred .pth (key 'model')")
    p.add_argument("--arch", type=str, default="pedpred3", choices=("pedpred", "pedpred2", "pedpred3"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("PEDPRED_BATCH", "16")),
        help="DataLoader batch (default: env PEDPRED_BATCH or 16)",
    )
    p.add_argument(
        "--nout",
        type=int,
        default=int(os.environ.get("PEDPRED_NOUT", "1")),
        help="Target horizon (default: env PEDPRED_NOUT or 1)",
    )
    p.add_argument(
        "--nin",
        type=int,
        default=None,
        help="History length T_hist (default: env PEDPRED_NIN or 5)",
    )
    p.add_argument(
        "--obs-mode",
        type=str,
        default="sensor",
        choices=("sensor", "sensor_static", "random"),
        help="sensor: PA-aligned partial-obs (MultiAgent+move_agents), not EnKF filter; sensor_static / random: ablations",
    )
    p.add_argument(
        "--sensing-range",
        type=float,
        default=5.0,
        help="Grid Euclidean radius (same as ENKF sensing_range); used when obs-mode=sensor",
    )
    p.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of agents (disk union); ENKF-style joint observation when >1",
    )
    p.add_argument(
        "--partial-frac",
        type=float,
        default=0.35,
        help="Fraction of cells observed; used only when obs-mode=random",
    )
    p.add_argument("--n-iter", type=int, default=8, help="Variational solver iterations")
    p.add_argument("--w-prior", type=float, default=0.5)
    p.add_argument(
        "--rho-mask-thr",
        type=float,
        default=0.05,
        help="真值密度 > 该阈值视为「有密度」格点；loss/MSE 与 prior 项仅在这些格上平均（与观测项 vel 掩码一致）",
    )
    p.add_argument("--use-gru", action="store_true", help="Use GRU spatial solver")
    p.add_argument("--train-workers", type=int, default=0)
    p.add_argument("--val-workers", type=int, default=0)
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, writes last.pt / best.pt (val) each epoch + training_meta.json",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Print train loss every N batches (1=every batch; 0=only epoch summary)",
    )
    if argv is None:
        argv = sys.argv[1:]
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)

    nin = int(args.nin if args.nin is not None else os.environ.get("PEDPRED_NIN", "5"))

    train_loader, val_loader = get_atc_data(
        "train",
        "valid",
        batch=args.batch,
        nin=nin,
        nout=args.nout,
        num_workers=args.train_workers,
        validation_num_workers=args.val_workers,
    )

    train_loader = wrap_loader_varnet(
        train_loader,
        seed=0,
        obs_mode=args.obs_mode,
        partial_frac=args.partial_frac,
        sensing_range=args.sensing_range,
        num_agents=args.num_agents,
        batch_size=args.batch,
        num_workers=args.train_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = wrap_loader_varnet(
        val_loader,
        seed=1,
        obs_mode=args.obs_mode,
        partial_frac=args.partial_frac,
        sensing_range=args.sensing_range,
        num_agents=args.num_agents,
        batch_size=args.batch,
        num_workers=args.val_workers,
        shuffle=False,
        drop_last=False,
    )

    print(
        f"[crowd_varnet] train_batches={len(train_loader)} val_batches={len(val_loader)}",
        flush=True,
    )

    ped = load_frozen_pedpred(args.checkpoint, device, arch=args.arch)
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=int(nin),
        n_iter=args.n_iter,
        use_gru=args.use_gru,
        w_prior=args.w_prior,
        rho_mask_thr=float(args.rho_mask_thr),
    ).to(device)

    opt = Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    print(
        f"[crowd_varnet] nin={nin} batch={args.batch} epochs={args.epochs} "
        f"obs_mode={args.obs_mode} save_dir={args.save_dir!r}",
        flush=True,
    )

    save_dir: Optional[Path] = Path(args.save_dir).resolve() if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        meta: Dict[str, Any] = {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "pedpred_ckpt": str(Path(args.checkpoint).resolve()),
            "arch": args.arch,
            "epochs": args.epochs,
            "nin": int(nin),
            "batch": int(args.batch),
            "nout": int(args.nout),
            "obs_mode": args.obs_mode,
            "sensing_range": args.sensing_range,
            "num_agents": args.num_agents,
            "n_iter": args.n_iter,
            "w_prior": args.w_prior,
            "log_interval": int(args.log_interval),
            "rho_mask_thr": float(args.rho_mask_thr),
        }
        (save_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_val = float("inf")

    for epoch in range(args.epochs):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            log_interval=args.log_interval,
            epoch=epoch,
            num_epochs=args.epochs,
        )
        model.eval()
        va = 0.0
        nv = 0
        for batch in val_loader:
            history, obs, obs_mask, x_gt = [b.to(device) for b in batch]
            loss, _ = model.compute_loss(history, obs, obs_mask, x_gt)
            va += float(loss.item())
            nv += 1
        model.train()
        va_mean = va / max(nv, 1)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={tr:.6f}  val_loss={va_mean:.6f}", flush=True)

        if save_dir is not None:
            payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": tr,
                "val_loss": va_mean,
            }
            torch.save(payload, save_dir / "last.pt")
            if va_mean < best_val:
                best_val = va_mean
                torch.save(payload, save_dir / "best.pt")
                print(f"  saved best.pt (val_loss={best_val:.6f}) -> {save_dir}", flush=True)


if __name__ == "__main__":
    main()
