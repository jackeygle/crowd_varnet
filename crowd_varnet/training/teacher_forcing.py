"""教师强制训练循环：history 取 GT，单帧 loss。"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..models.varnet import CrowdVarNet


def train_one_epoch(
    model: CrowdVarNet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    log_interval: int = 1,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    eprefix = (
        f"[{int(epoch) + 1}/{int(num_epochs)}] "
        if epoch is not None and num_epochs is not None
        else ""
    )
    n_batches = len(loader)
    for batch in loader:
        history, obs, obs_mask, x_gt = batch
        history = history.to(device)
        obs = obs.to(device)
        obs_mask = obs_mask.to(device)
        x_gt = x_gt.to(device)

        loss, info = model.compute_loss(history, obs, obs_mask, x_gt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
        n += 1
        if log_interval > 0 and (n % log_interval == 0 or n == n_batches):
            print(
                f"{eprefix}batch {n}/{n_batches}  loss={loss.item():.6f}"
                f"  rho={info['rho_mse']:.4f}  vx={info['vx_mse']:.4f}  vy={info['vy_mse']:.4f}",
                flush=True,
            )
    return total / max(n, 1)
