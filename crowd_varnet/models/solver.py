"""可训练的迭代求解器（VarNet 核心）。

三种 solver 实现：
  - ``scalar`` (旧 ``use_gru=False``)：每步一个标量步长，仅 ``n_iter`` 个参数；
  - ``gru``    (旧 ``use_gru=True``) ：``GRUCell`` per-pixel 更新，~1.3k 参数；
  - ``convgru`` (新)                  ：``ConvGRU`` 共享空间核 + 输出头，
                                       ~25k–300k 参数（依 hidden）；显式利用空间相邻信息。

旧 ckpt（``log_steps`` 或 ``solver.gru_cell.*``）仍可加载——
``CrowdVarNet`` 的 ``cli._common.build_model_from_ckpt`` 会从 state_dict 键名自动推断。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cost import VariationalCost, clip_crowd_state


class ConvGRUCell(nn.Module):
    """规范 ConvGRU：用两次卷积分别算 (r,z) 与 n。

    参数量 ≈ ``9 * (in+h) * (3h)``，相比 per-pixel ``GRUCell`` 多了空间核但远少于 U-Net。
    """

    def __init__(self, in_ch: int, hidden: int, k: int = 3):
        super().__init__()
        self.hidden = hidden
        pad = k // 2
        self.conv_rz = nn.Conv2d(in_ch + hidden, 2 * hidden, kernel_size=k, padding=pad)
        self.conv_n = nn.Conv2d(in_ch + hidden, hidden, kernel_size=k, padding=pad)
        for m in (self.conv_rz, self.conv_n):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        rz = self.conv_rz(torch.cat([x, h], dim=1))
        r, z = rz.chunk(2, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(self.conv_n(torch.cat([x, r * h], dim=1)))
        return (1 - z) * n + z * h


class _OutHead(nn.Module):
    """ConvGRU hidden ``[B,h,H,W]`` → 4 通道 δ：两层 Conv3x3 + GN + ReLU，再 1x1 出 4。

    Tanh 软限幅避免单步过大。可选在中间插 Dropout2d 做 regularization。
    """

    def __init__(self, hidden: int, mid: Optional[int] = None, dropout_p: float = 0.0):
        super().__init__()
        m = mid if mid is not None else hidden
        layers = [
            nn.Conv2d(hidden, m, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, m), num_channels=m),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers += [
            nn.Conv2d(m, m, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, m), num_channels=m),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))
        layers.append(nn.Conv2d(m, 4, 1))
        self.body = nn.Sequential(*layers)
        # 末层零初始化：训练初期 δ≈0，恒等于 LISTA 起点（x_init），稳定起步。
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.scale = nn.Parameter(torch.ones(4) * 0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.body(h))
        return delta * self.scale.view(1, 4, 1, 1)


class CrowdVarNetIterativeSolver(nn.Module):
    """在格点状态 ``x`` 上对 ``VariationalCost`` 做多步下降。

    ``solver_type``：
      - ``scalar``  : 每步一个可学习标量步长（``log_steps``），向后兼容旧 ckpt；
      - ``gru``     : per-pixel ``GRUCell``（旧 ``use_gru=True``）；
      - ``convgru`` : 共享空间核 ConvGRU + 输出头（**推荐**）。

    旧用法 ``CrowdVarNetIterativeSolver(n_iter, use_gru=True)`` 仍工作。
    """

    def __init__(
        self,
        n_iter: int = 8,
        use_gru: bool = False,
        gru_ch: int = 16,
        clip_each_step: bool = True,
        *,
        solver_type: Optional[str] = None,
        hidden: int = 32,
        kernel: int = 3,
        share_across_iter: bool = True,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.clip_each_step = clip_each_step
        self.gru_ch = gru_ch
        self.hidden = hidden
        self.dropout_p = dropout_p

        if solver_type is None:
            solver_type = "gru" if use_gru else "scalar"
        solver_type = solver_type.lower()
        if solver_type not in ("scalar", "gru", "convgru"):
            raise ValueError(f"solver_type must be scalar/gru/convgru, got {solver_type!r}")
        self.solver_type = solver_type

        if solver_type == "scalar":
            self.log_steps = nn.Parameter(torch.full((n_iter,), -2.0))
        elif solver_type == "gru":
            self.gru_cell = nn.GRUCell(input_size=8, hidden_size=gru_ch)
            self.gru_out = nn.Sequential(nn.Linear(gru_ch, 4), nn.Tanh())
        else:  # convgru
            # 输入特征 [grad(4), x(4), x_prior(4), obs(4), obs_mask(1)] = 17
            in_ch = 17
            if share_across_iter:
                self.convgru = ConvGRUCell(in_ch=in_ch, hidden=hidden, k=kernel)
                self.out_head = _OutHead(hidden, dropout_p=dropout_p)
                self._convgru_list = None
                self._head_list = None
            else:
                self.convgru = None
                self.out_head = None
                self._convgru_list = nn.ModuleList(
                    [ConvGRUCell(in_ch=in_ch, hidden=hidden, k=kernel) for _ in range(n_iter)]
                )
                self._head_list = nn.ModuleList([_OutHead(hidden, dropout_p=dropout_p) for _ in range(n_iter)])
            self.share_across_iter = share_across_iter

    def _step_convgru(
        self, k: int, x: torch.Tensor, grad: torch.Tensor,
        x_prior: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = torch.cat([grad, x, x_prior, obs, obs_mask], dim=1)
        if self.share_across_iter:
            h = self.convgru(feat, h)
            delta = self.out_head(h)
        else:
            assert self._convgru_list is not None and self._head_list is not None
            h = self._convgru_list[k](feat, h)
            delta = self._head_list[k](h)
        return delta, h

    def forward(
        self,
        x_init: torch.Tensor,
        cost_fn: VariationalCost,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_prior: torch.Tensor,
    ) -> torch.Tensor:
        x = x_init.clone().detach().requires_grad_(True)
        B, C, H, W = x.shape

        if self.solver_type == "gru":
            h = torch.zeros(B * H * W, self.gru_ch, device=x.device, dtype=x.dtype)
        elif self.solver_type == "convgru":
            h = torch.zeros(B, self.hidden, H, W, device=x.device, dtype=x.dtype)
        else:
            h = None

        for k in range(self.n_iter):
            cost, _, _ = cost_fn(x, obs, obs_mask, x_prior)
            grad = torch.autograd.grad(cost, x, create_graph=True)[0]

            if self.solver_type == "scalar":
                x = x - torch.exp(self.log_steps[k]) * grad
            elif self.solver_type == "gru":
                assert h is not None
                feat = torch.cat([grad, x.detach()], dim=1)
                feat_flat = feat.permute(0, 2, 3, 1).reshape(B * H * W, C * 2)
                h = self.gru_cell(feat_flat, h)
                delta = self.gru_out(h).reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = x - delta
            else:  # convgru
                assert h is not None
                delta, h = self._step_convgru(k, x, grad, x_prior, obs, obs_mask, h)
                x = x - delta

            if self.clip_each_step:
                x = clip_crowd_state(x)
            x = x.requires_grad_(True)

        return x
