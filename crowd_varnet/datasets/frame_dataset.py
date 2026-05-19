"""单帧 dataset：从 SeqDataset 取一个目标帧 + 5 帧 GT history，生成部分观测。"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from ..deps.enkf_sensors import MultiAgent
from .sensors import (
    spatial_sensor_mask,
    stack_grid_sequence,
    target_frame_index_in_episode,
)


class CrowdVarNetDataset(torch.utils.data.Dataset):
    """
    从 ``SeqDataset`` 取 history / x_gt，并构造部分观测。

    **定位**：CrowdVarNet 学的是「部分观测 + PedPred 先验 → 重建全场」的变分网络，与 EnKF 对齐的只是
    实验配置（``MultiAgent`` goal/``move_agents``、``num_agents``、``sensing_range``、圆盘并集几何），
    不把 EnKF 算法并入本模型。

    **模式**：
    - ``obs_mode="sensor"``（默认）：每个 episode 固定 RNG 初始化 agent，对目标帧下标 ``G`` 执行 ``G+1``
      次 ``move_agents()`` 再算 mask（与 PA 里"先动再采观测"的步序一致）。
    - ``obs_mode="sensor_static"``：每样本独立随机圆心（不做 goal 运动；消融用）。
    - ``obs_mode="random"``：随机格子比例 ``partial_frac``（消融用）。
    """

    def __init__(
        self,
        seq_ds: torch.utils.data.Dataset,
        *,
        obs_mode: str = "sensor",
        partial_frac: float = 0.35,
        sensing_range: float = 5.0,
        num_agents: int = 3,
        seed: int = 0,
    ):
        self.seq_ds = seq_ds
        self.obs_mode = obs_mode.lower().strip()
        if self.obs_mode not in ("sensor", "sensor_static", "random"):
            raise ValueError(
                f"obs_mode must be 'sensor', 'sensor_static', or 'random', got {self.obs_mode!r}"
            )
        self.partial_frac = partial_frac
        self.sensing_range = float(sensing_range)
        self.num_agents = max(1, int(num_agents))
        self._seed = int(seed)

    def __len__(self) -> int:
        return len(self.seq_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp, tgt = self.seq_ds[idx]
        history = stack_grid_sequence(inp)
        x_gt = stack_grid_sequence(tgt)[0]

        _, _, H, W = history.shape

        if self.obs_mode == "sensor":
            G = target_frame_index_in_episode(self.seq_ds, idx)
            rng_agents = np.random.RandomState(int(self._seed) + idx * 100003)
            agents_ma = MultiAgent(
                (H, W),
                (4, H, W),
                sensing_range=self.sensing_range,
                num_agents=self.num_agents,
                rng=rng_agents,
            )
            for _ in range(G + 1):
                agents_ma.move_agents()
            obs_mask = spatial_sensor_mask(
                H, W, list(agents_ma.positions), float(self.sensing_range)
            )
        elif self.obs_mode == "sensor_static":
            rng = np.random.RandomState(self._seed + idx * 100003)
            agents = [
                (int(rng.randint(0, H)), int(rng.randint(0, W)))
                for _ in range(self.num_agents)
            ]
            obs_mask = spatial_sensor_mask(H, W, agents, self.sensing_range)
        else:
            flat = H * W
            n_vis = max(1, int(flat * self.partial_frac))
            g = torch.Generator()
            g.manual_seed(self._seed + idx)
            perm = torch.randperm(flat, generator=g)
            vis_idx = perm[:n_vis]
            obs_mask = torch.zeros(1, H, W)
            obs_mask.reshape(-1)[vis_idx] = 1.0

        obs = x_gt * obs_mask
        return history, obs, obs_mask, x_gt
