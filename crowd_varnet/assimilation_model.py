"""
兼容 shim：主实现已拆到 ``models/``, ``datasets/``, ``training/`` 子包。

新代码请直接从子包 import：
  - ``from crowd_varnet.models import CrowdVarNet, VariationalCost, ...``
  - ``from crowd_varnet.datasets import CrowdVarNetDataset, RolloutEpisodeDataset, ...``
  - ``from crowd_varnet.training import train_one_epoch, rollout_tbptt_epoch, ...``

旧代码 ``from crowd_varnet.assimilation_model import CrowdVarNet`` 等写法继续有效。
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import torch
import torch.nn as nn

from .datasets import (
    CrowdVarNetDataset,
    RolloutEpisodeDataset,
    spatial_sensor_mask,
    stack_grid_sequence,
    target_frame_index_in_episode,
    unwrap_concat_base_dataset,
)
from .models import (
    CrowdVarNet,
    CrowdVarNetIterativeSolver,
    FrozenPedPredPrior,
    VariationalCost,
    clip_crowd_state,
    density_support_mask,
    load_frozen_pedpred,
    masked_mean_sq,
)
from .training import rollout_tbptt_epoch, rollout_val_loss, train_one_epoch

# 旧名别名（checkpoint 子模块名 ``adapter`` / ``solver`` 不受影响）
PedPredAdapter = FrozenPedPredPrior
LearnedGradSolver = CrowdVarNetIterativeSolver


def parse_smoke_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CrowdVarNet smoke")
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def run_smoke() -> None:
    args = parse_smoke_args()
    device = torch.device(args.device)
    B, T, H, W = 2, 5, 36, 12

    class DummyPedPred(nn.Module):
        def forward(self, inp, hidden=None, *, horizon=1):
            t = inp if torch.is_tensor(inp) else torch.as_tensor(inp)
            if t.dim() == 5:
                t = t[:, -1]
            return t.unsqueeze(1)

    model = CrowdVarNet(
        DummyPedPred().to(device),
        freeze_phi=True,
        T_hist=T,
        n_iter=6,
        use_gru=False,
    ).to(device)

    history = torch.rand(B, T, 4, H, W, device=device)
    obs_mask = torch.zeros(B, 1, H, W, device=device)
    obs_mask[:, :, :, : W // 2] = 1.0
    obs = torch.rand(B, 4, H, W, device=device) * obs_mask
    x_gt = torch.rand(B, 4, H, W, device=device)

    loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
    print("loss", loss.item(), "phi_mse", info["phi_mse"], "recon", info["recon"])


__all__ = [
    "CrowdVarNet",
    "CrowdVarNetDataset",
    "CrowdVarNetIterativeSolver",
    "FrozenPedPredPrior",
    "LearnedGradSolver",
    "PedPredAdapter",
    "RolloutEpisodeDataset",
    "VariationalCost",
    "clip_crowd_state",
    "density_support_mask",
    "load_frozen_pedpred",
    "masked_mean_sq",
    "rollout_tbptt_epoch",
    "rollout_val_loss",
    "run_smoke",
    "spatial_sensor_mask",
    "stack_grid_sequence",
    "target_frame_index_in_episode",
    "train_one_epoch",
    "unwrap_concat_base_dataset",
]
