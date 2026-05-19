"""Compare rollout baselines on the same val loader as training.

Modes:
  pedpred   : pure open-loop forecast, x_hat = ped_pred(history); ignores observations.
  naive     : x_hat = obs * obs_mask + ped_pred(history) * (1 - obs_mask) (the solver's x_init).
  cvn       : full CrowdVarNet rollout (same as training-time val).

For each mode we report per-channel masked MSE so it is directly comparable to
the per-channel numbers logged during training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader

from .deps.dataset_atc import get_atc_data
from .datasets import RolloutEpisodeDataset, unwrap_concat_base_dataset
from .models import CrowdVarNet, load_frozen_pedpred
from .models.cost import density_support_mask, masked_mean_sq
from .models.prior import FrozenPedPredPrior


_CH_NAMES = ("rho", "vx", "vy", "speed")
_CH_W = (2.5, 1.5, 1.0, 0.5)


def _build_val_loader(args, T_hist: int) -> DataLoader:
    _, base_val = get_atc_data(
        "train", "valid",
        batch=1, nin=T_hist, nout=1,
        num_workers=0, validation_num_workers=0,
    )
    bases = list(unwrap_concat_base_dataset(base_val.dataset))
    wrapped = [
        RolloutEpisodeDataset(
            b, episode_len=args.episode_len,
            sensing_range=args.sensing_range, num_agents=args.num_agents,
            seed=113 + i * 7919,
        )
        for i, b in enumerate(bases)
    ]
    ds = wrapped[0] if len(wrapped) == 1 else ConcatDataset(wrapped)
    return DataLoader(
        dataset=ds,
        batch_size=max(1, args.batch // 2),
        shuffle=False, num_workers=0, drop_last=False,
    )


@torch.no_grad()
def _rollout_eval(
    model: Optional[CrowdVarNet],
    prior: FrozenPedPredPrior,
    loader: DataLoader,
    device: torch.device,
    *,
    mode: str,
    warmup: int,
    max_episodes: int,
    rho_thr: float,
) -> Dict[str, float]:
    ch_w = torch.tensor(_CH_W, dtype=torch.float32, device=device).view(1, 4, 1, 1)
    loss_sum = 0.0
    steps = 0
    ch_sq_sum = torch.zeros(4, device=device)
    n_mask_sum = torch.zeros((), device=device)

    if model is not None:
        for p in model.parameters():
            p.requires_grad_(False)

    n_seen = 0
    for batch in loader:
        x_gt_seq, obs_seq, mask_seq = batch
        x_gt_seq = x_gt_seq.to(device)
        obs_seq = obs_seq.to(device)
        mask_seq = mask_seq.to(device)
        B, L, C, H, W = x_gt_seq.shape
        history_buf = x_gt_seq[:, :warmup].clone()

        for t in range(warmup, L):
            if mode == "pedpred":
                x_hat = prior(history_buf).detach()
            elif mode == "naive":
                x_prior = prior(history_buf).detach()
                x_hat = obs_seq[:, t] * mask_seq[:, t] + x_prior * (1.0 - mask_seq[:, t])
            elif mode == "cvn":
                assert model is not None
                with torch.enable_grad():
                    x_hat = model.forward(history_buf, obs_seq[:, t], mask_seq[:, t]).detach()
            else:
                raise ValueError(f"unknown mode={mode}")

            err = (x_hat - x_gt_seq[:, t]).pow(2)
            dm = density_support_mask(x_gt_seq[:, t], rho_thr)
            m4 = dm.expand_as(x_gt_seq[:, t])
            step_loss = masked_mean_sq(err * ch_w, m4)
            loss_sum += float(step_loss.item())
            steps += 1
            ch_sq_sum += (err * dm).sum(dim=(0, 2, 3))
            n_mask_sum += dm.sum()
            history_buf = torch.cat([history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1)

        n_seen += B
        if n_seen >= max_episodes:
            break

    n_safe = n_mask_sum.clamp_min(1.0)
    ch_mse = (ch_sq_sum / n_safe).cpu().tolist()
    out = {"total": loss_sum / max(steps, 1)}
    for i, name in enumerate(_CH_NAMES):
        out[name] = float(ch_mse[i])
        out[f"{name}_w"] = float(_CH_W[i] * ch_mse[i] / 4.0)
    return out


def _fmt(comp: Dict[str, float], tag: str) -> str:
    return (
        f"{tag}: total={comp['total']:.4f}  "
        f"rho={comp['rho']:.4f} vx={comp['vx']:.4f} "
        f"vy={comp['vy']:.4f} sp={comp['speed']:.4f} | "
        f"w*: rho={comp['rho_w']:.4f} vx={comp['vx_w']:.4f} "
        f"vy={comp['vy_w']:.4f} sp={comp['speed_w']:.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="PedPred ckpt (.hkl/.pth)")
    ap.add_argument("--arch", default="pedpred3")
    ap.add_argument("--cvn-ckpt", default=None, help="best.pt (CrowdVarNet) for cvn mode")
    ap.add_argument("--modes", nargs="+",
                    default=["pedpred", "naive"],
                    choices=["pedpred", "naive", "cvn"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--episode-len", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sensing-range", type=float, default=5.0)
    ap.add_argument("--num-agents", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--use-gru", action="store_true")
    ap.add_argument("--gru-ch", type=int, default=16)
    ap.add_argument("--w-prior", type=float, default=0.5)
    ap.add_argument("--rho-mask-thr", type=float, default=0.05)
    ap.add_argument("--max-episodes", type=int, default=16)
    ap.add_argument("--save-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ped = load_frozen_pedpred(args.checkpoint, device, arch=args.arch)
    prior = FrozenPedPredPrior(ped, freeze=True).to(device).eval()

    val_loader = _build_val_loader(args, T_hist=args.warmup)
    print(f"[eval] val_episodes={len(val_loader.dataset)} max_episodes={args.max_episodes}", flush=True)

    cvn_model: Optional[CrowdVarNet] = None
    if "cvn" in args.modes:
        cvn_model = CrowdVarNet(
            ped_pred=ped, freeze_phi=True, T_hist=args.warmup,
            n_iter=args.n_iter, use_gru=args.use_gru, gru_ch=args.gru_ch,
            w_prior=args.w_prior, rho_mask_thr=args.rho_mask_thr,
        ).to(device)
        if args.cvn_ckpt:
            sd = torch.load(args.cvn_ckpt, map_location=device)
            sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
            missing, unexpected = cvn_model.load_state_dict(sd, strict=False)
            print(f"[eval] loaded cvn ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    results: Dict[str, Dict[str, float]] = {}
    for mode in args.modes:
        comp = _rollout_eval(
            cvn_model if mode == "cvn" else None,
            prior, val_loader, device,
            mode=mode, warmup=args.warmup,
            max_episodes=args.max_episodes,
            rho_thr=args.rho_mask_thr,
        )
        results[mode] = comp
        print(_fmt(comp, f"{mode:7s}"), flush=True)

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[eval] saved -> {args.save_json}", flush=True)


if __name__ == "__main__":
    main()
