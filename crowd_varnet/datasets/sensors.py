"""部分观测的几何与序列工具。"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from ..deps.grid_data import GridData


def spatial_sensor_mask(
    H: int,
    W: int,
    agents_rc: Sequence[Tuple[int, int]],
    sensing_range: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    部分观测的**几何**与 PA/ENKF 脚本里 ``GeneratePartialObs.cells_within_range`` 相同（实验配置一致）：
    格点 (r,c) 满足 ``sqrt((r-r0)^2 + (c-c0)^2) <= sensing_range`` 视为可见；多智能体圆盘并集。
    返回 ``obs_mask`` 形状 ``[1, H, W]``。
    """
    rr = torch.arange(H, dtype=dtype, device=device).view(H, 1)
    cc = torch.arange(W, dtype=dtype, device=device).view(1, W)
    mask = torch.zeros(H, W, dtype=dtype, device=device)
    sr = float(sensing_range)
    sr2 = sr * sr
    for r0, c0 in agents_rc:
        dist_sq = (rr - float(r0)) ** 2 + (cc - float(c0)) ** 2
        mask = torch.maximum(mask, (dist_sq <= sr2).to(dtype))
    return mask.unsqueeze(0)


def target_frame_index_in_episode(
    seq_ds: torch.utils.data.Dataset, local_window_idx: int
) -> int:
    """H5 内目标帧下标（与 ``SeqDataset`` 的 ``start + input_len`` 一致）。"""
    if not hasattr(seq_ds, "input_len") or not hasattr(seq_ds, "step"):
        raise TypeError("seq_ds must be a SeqDataset (input_len, step).")
    step = seq_ds.step
    if callable(step):
        step = int(step())
    else:
        step = int(step)
    return int(local_window_idx * step + int(seq_ds.input_len))


def stack_grid_sequence(seq: List[Any]) -> torch.Tensor:
    tensors = []
    for f in seq:
        if isinstance(f, GridData) or hasattr(f, "as_tensor"):
            tensors.append(GridData(f).as_tensor("density", "vel_mean", "vel_var"))
        else:
            tensors.append(torch.as_tensor(f))
    return torch.stack(tensors, dim=0)
