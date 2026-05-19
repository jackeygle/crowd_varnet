"""推理时 n_iter 扫描：固定 ckpt，改变 solver.n_iter，看 val rollout loss + 部分 diag。

不需要重训。直接读 best.pt，patch ``model.solver.n_iter``，跑 val rollout。

用法::
    python -m scripts.sweep_n_iter --ckpt /path/to/best.pt --n-iters 1 2 4 6 8 10 12 16
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from crowd_varnet.cli import build_model_from_ckpt
from crowd_varnet.deps.dataset_atc import get_atc_data
from crowd_varnet.datasets import RolloutEpisodeDataset, unwrap_concat_base_dataset
from crowd_varnet.training import rollout_val_loss


def _build_val_loader(meta: dict, val_workers: int):
    T_hist = int(meta.get("nin", 5))
    sensing_range = float(meta.get("sensing_range", 5.0))
    num_agents = int(meta.get("num_agents", 3))
    episode_len = int(meta.get("episode_len", 300))
    batch = max(1, int(meta.get("batch", 16)) // 2)

    _, base_val_loader = get_atc_data(
        "train", "valid", batch=1, nin=T_hist, nout=1,
        num_workers=0, validation_num_workers=0,
    )
    bases = list(unwrap_concat_base_dataset(base_val_loader.dataset))
    wrapped = [
        RolloutEpisodeDataset(
            b, episode_len=episode_len,
            sensing_range=sensing_range, num_agents=num_agents,
            seed=113 + i * 7919,
        ) for i, b in enumerate(bases)
    ]
    ds = wrapped[0] if len(wrapped) == 1 else ConcatDataset(wrapped)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=val_workers,
                      drop_last=False), int(meta.get("warmup", T_hist))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-iters", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10, 12, 16])
    ap.add_argument("--val-max-episodes", type=int, default=64)
    ap.add_argument("--val-workers", type=int, default=2)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt).resolve()
    model, meta = build_model_from_ckpt(ckpt_path, device=device)
    model.eval()
    val_loader, warmup = _build_val_loader(meta, args.val_workers)

    # cap iterator
    capped = []
    batch = max(1, int(meta.get("batch", 16)) // 2)
    for i, b in enumerate(val_loader):
        if i * batch >= args.val_max_episodes:
            break
        capped.append(b)

    class _ListLoader:
        def __init__(self, items): self.items = items
        def __iter__(self): return iter(self.items)
        def __len__(self): return len(self.items)

    print(f"[sweep] ckpt={ckpt_path.name} train_n_iter={model.solver.n_iter} "
          f"sweep over {args.n_iters}  val_episodes={len(capped)*batch}", flush=True)

    results = []
    print(f"\n{'n_iter':>6} | {'total':>7} {'rho':>7} {'vx':>7} {'vy':>7} {'speed':>7} | {'rho_bg_mse':>10} {'rho_bg_mean':>11} | {'time_s':>7}")
    for n in args.n_iters:
        model.solver.n_iter = int(n)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.time()
        out = rollout_val_loss(model, _ListLoader(capped), device, warmup=warmup)
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.time() - t0
        out["n_iter"] = int(n)
        out["time_s"] = dt
        results.append(out)
        print(f"{n:>6} | {out['total']:>7.4f} {out['rho']:>7.4f} {out['vx']:>7.4f} "
              f"{out['vy']:>7.4f} {out['speed']:>7.4f} | "
              f"{out.get('rho_bg_mse', float('nan')):>10.6f} "
              f"{out.get('rho_bg_mean', float('nan')):>11.6f} | "
              f"{dt:>7.1f}", flush=True)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[sweep] best n_iter = {min(results, key=lambda r: r['total'])['n_iter']}", flush=True)


if __name__ == "__main__":
    main()
