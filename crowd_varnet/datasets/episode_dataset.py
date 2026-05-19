"""全 episode dataset：切 N 步连续帧，用于 rollout / TBPTT 训练。"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..deps.enkf_sensors import MultiAgent
from ..deps.grid_data import GridData
from .sensors import spatial_sensor_mask


class RolloutEpisodeDataset(torch.utils.data.Dataset):
    """
    把连续 H5 序列切成长度 ``episode_len`` 的 episode，用于 rollout / TBPTT 训练。

    每个 ``__getitem__`` 返回一整段 episode：
      - ``x_gt_seq``:   [L, 4, H, W]  连续真值
      - ``obs_seq``:    [L, 4, H, W]  ``x_gt * obs_mask``
      - ``obs_mask_seq``: [L, 1, H, W] 传感器圆盘并集（每步 ``move_agents`` 一次）

    只能包 "按 index 取单帧" 的底层 dataset（如 ``CachedGridH5Dataset``）；不支持 ``SeqDataset``。
    Agent 初始 seed 由 ``(self._seed + idx * 100003)`` 派生，保证每个 episode 几何不同。
    """

    def __init__(
        self,
        base_ds: torch.utils.data.Dataset,
        *,
        episode_len: int = 300,
        sensing_range: float = 5.0,
        num_agents: int = 3,
        seed: int = 0,
        flip_w_prob: float = 0.0,
        obs_noise_std: Optional[Sequence[float]] = None,
    ):
        self.base_ds = base_ds
        self.episode_len = int(episode_len)
        self.sensing_range = float(sensing_range)
        self.num_agents = max(1, int(num_agents))
        self._seed = int(seed)
        self.flip_w_prob = float(flip_w_prob)
        if obs_noise_std is not None:
            arr = [float(x) for x in obs_noise_std]
            assert len(arr) == 4, "obs_noise_std 必须是 4 元（rho, vx, vy, var）"
            self.obs_noise_std = arr
        else:
            self.obs_noise_std = None
        L = len(base_ds)
        self._num_episodes = L // self.episode_len

    def __len__(self) -> int:
        return self._num_episodes

    def __getitem__(self, idx: int):
        start = idx * self.episode_len
        L = self.episode_len

        frames = []
        for t in range(L):
            f = self.base_ds[start + t]
            if isinstance(f, GridData) or hasattr(f, "as_tensor"):
                frames.append(GridData(f).as_tensor("density", "vel_mean", "vel_var"))
            else:
                frames.append(torch.as_tensor(f))
        x_gt_seq = torch.stack(frames, dim=0)  # [L, 4, H, W]

        _, _, H, W = x_gt_seq.shape
        rng = np.random.RandomState(self._seed + idx * 100003)
        agents = MultiAgent(
            (H, W),
            (4, H, W),
            sensing_range=self.sensing_range,
            num_agents=self.num_agents,
            rng=rng,
        )

        mask_seq = torch.empty(L, 1, H, W, dtype=x_gt_seq.dtype)
        for t in range(L):
            agents.move_agents()
            mask_seq[t] = spatial_sensor_mask(
                H, W, list(agents.positions), self.sensing_range, dtype=x_gt_seq.dtype
            )

        obs_seq = x_gt_seq * mask_seq

        # 训练增强：在 obs 上加高斯噪声（mask 内才生效，因为掩码外被 mask 强制为 0）。
        # σ 是 4 通道独立标定（rho, vx, vy, var）。
        if self.obs_noise_std is not None:
            std_t = torch.tensor(self.obs_noise_std, dtype=obs_seq.dtype).view(1, 4, 1, 1)
            noise = torch.randn(L, 4, H, W, dtype=obs_seq.dtype) * std_t
            obs_seq = obs_seq + noise * mask_seq

        # 数据增强：以 ``flip_w_prob`` 概率沿 W 轴（最后一维）整段翻转，且 vx 取反。
        # 对 ATC corridor 这条对称轴有效（人左右走的频率近似对称）。
        # 整段一起翻，保持时序一致；mask 也一起翻（圆盘对称，等价于 agent 位置翻）。
        if self.flip_w_prob > 0.0:
            if rng.rand() < self.flip_w_prob:
                x_gt_seq = torch.flip(x_gt_seq, dims=(-1,))
                x_gt_seq[:, 1] = -x_gt_seq[:, 1]    # vx 取反
                obs_seq = torch.flip(obs_seq, dims=(-1,))
                obs_seq[:, 1] = -obs_seq[:, 1]
                mask_seq = torch.flip(mask_seq, dims=(-1,))
        return x_gt_seq, obs_seq, mask_seq


def unwrap_concat_base_dataset(ds):
    """提取 ConcatDataset 里每个底层 dataset（跨 SeqDataset / ConcatDataset 层），
    用于把 SeqDataset 的 ConcatDataset 重打包成 episode 级 dataset。"""
    from torch.utils.data import ConcatDataset

    if isinstance(ds, ConcatDataset):
        for sub in ds.datasets:
            yield from unwrap_concat_base_dataset(sub)
        return
    inner = getattr(ds, "dataset", None)
    if inner is not None and not isinstance(inner, (list, tuple)):
        yield from unwrap_concat_base_dataset(inner)
        return
    yield ds
