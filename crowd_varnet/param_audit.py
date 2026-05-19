"""Print CrowdVarNet trainable parameter audit (total / by submodule)."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

from .models import CrowdVarNet, load_frozen_pedpred


def _count(params) -> int:
    return sum(p.numel() for p in params)


def _audit(model: torch.nn.Module) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, mod in model.named_children():
        n_total = _count(mod.parameters())
        n_train = _count(p for p in mod.parameters() if p.requires_grad)
        out[name] = {"total": n_total, "trainable": n_train}
    out["__overall__"] = {
        "total": _count(model.parameters()),
        "trainable": _count(p for p in model.parameters() if p.requires_grad),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--arch", type=str, default="pedpred3")
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument("--use-gru", action="store_true", help="[Deprecated] 等价 --solver-type gru")
    p.add_argument(
        "--solver-type",
        type=str,
        default=None,
        choices=("scalar", "gru", "convgru"),
    )
    p.add_argument("--solver-hidden", type=int, default=32)
    p.add_argument("--solver-no-share", action="store_true")
    p.add_argument("--init-gate", action="store_true")
    p.add_argument("--init-gate-mid", type=int, default=16)
    p.add_argument("--gru-ch", type=int, default=16)
    p.add_argument("--w-prior", type=float, default=0.5)
    p.add_argument("--rho-mask-thr", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ped = load_frozen_pedpred(args.checkpoint, dev, arch=args.arch)
    solver_type = args.solver_type
    if solver_type is None:
        solver_type = "gru" if args.use_gru else "scalar"
    model = CrowdVarNet(
        ped_pred=ped,
        freeze_phi=True,
        T_hist=args.warmup,
        n_iter=args.n_iter,
        use_gru=args.use_gru,
        gru_ch=args.gru_ch,
        w_prior=args.w_prior,
        rho_mask_thr=args.rho_mask_thr,
        solver_type=solver_type,
        solver_hidden=args.solver_hidden,
        solver_share=not args.solver_no_share,
        init_gate=args.init_gate,
        init_gate_mid=args.init_gate_mid,
    ).to(dev)

    info = _audit(model)
    print("=== CrowdVarNet parameter audit ===")
    for name, d in info.items():
        if name == "__overall__":
            continue
        print(f"  {name:20s} total={d['total']:>10,}  trainable={d['trainable']:>10,}")
    o = info["__overall__"]
    print(f"  {'OVERALL':20s} total={o['total']:>10,}  trainable={o['trainable']:>10,}")

    print("\n=== Trainable param tensors (top 20 by numel) ===")
    rows = [
        (name, p.numel(), tuple(p.shape))
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, n, shape in rows[:20]:
        print(f"  {n:>10,}  {str(shape):24s}  {name}")
    if len(rows) > 20:
        print(f"  (+ {len(rows) - 20} more trainable tensors)")
    print(f"\n  total trainable tensors: {len(rows)}")


if __name__ == "__main__":
    main()
