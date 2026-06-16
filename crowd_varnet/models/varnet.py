"""端到端 CrowdVarNet：冻结 PedPred 先验 + ConvGRU 迭代求解器。

可选输出 per-pixel log-variance（heteroscedastic NLLL，用于 Deep Ensemble 时
分解 aleatoric / epistemic uncertainty）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .cost import VariationalCost, density_support_mask, masked_mean_sq
from .prior import FrozenPedPredPrior
from .solver import CrowdVarNetIterativeSolver


class _VarHead(nn.Module):
    """Per-pixel log-variance head for ρ, vx, vy.

    输入是 solver 的 4 通道输出 ``x_hat``。我们故意不用 solver 内部的 hidden
    state，因为这样：
      - var_head 是独立的可控模块，关闭时只是不调用，不影响主路径
      - 训练 / 推理时跟主输出解耦，方便消融与冻结
    最后一层 zero-init → 训练初期 log_var ≈ 0 → σ² ≈ 1，避免起步爆炸。
    """

    def __init__(self, in_ch: int = 4, mid: int = 32, out_ch: int = 3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, mid), num_channels=mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, mid), num_channels=mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, kernel_size=1),
        )
        # zero-init last layer for stable start (log_var ≈ 0 → σ² ≈ 1)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        return self.body(x_hat)


class CrowdVarNet(nn.Module):
    """
    变分人群状态重建网络。

    给定历史帧 history、稀疏观测 obs 和观测掩码 obs_mask，
    利用冻结的 PedPred 先验和可学习的 ConvGRU 迭代求解器重建完整状态。

    训练与验证的 MSE 均在 x_gt 密度 > rho_mask_thr 的格点上平均。

    可选 ``predict_uncertainty=True``：额外预测 ρ/vx/vy 的 log-variance，用于
    异方差 NLLL 训练 + Deep Ensemble 推理时的不确定度分解。
    """

    def __init__(
        self,
        ped_pred: nn.Module,
        freeze_phi: bool = True,
        T_hist: int = 5,
        n_iter: int = 8,
        w_obs: float = 1.0,
        w_prior: float = 0.5,
        ch_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0),
        rho_mask_thr: float = 0.05,
        prior_use_ch_weights: bool = True,
        clip_solver_steps: bool = True,
        *,
        solver_hidden: int = 256,
        solver_kernel: int = 3,
        solver_share: bool = True,
        solver_dropout: float = 0.0,
        unfreeze_phi_tail: int = 0,
        predict_uncertainty: bool = False,
        log_var_clamp: Tuple[float, float] = (-7.0, 5.0),
        solver_use_attention: bool = True,
        solver_attn_heads: int = 4,
        solver_momentum: float = 0.5,
        solver_rnn_type: str = "gru",
        prior_unobs_weight: float = 0.05,
        solver_lr_grad: float = 0.0,
        solver_use_obs_encoder: bool = False,
    ):
        super().__init__()
        self.T_hist = T_hist
        self.adapter = FrozenPedPredPrior(
            ped_pred, freeze=freeze_phi, unfreeze_tail_layers=unfreeze_phi_tail
        )
        self.cost_fn = VariationalCost(
            w_obs,
            w_prior,
            ch_weights,
            rho_mask_thr,
            prior_use_ch_weights=prior_use_ch_weights,
            prior_unobs_weight=prior_unobs_weight,
        )
        self.solver = CrowdVarNetIterativeSolver(
            n_iter,
            hidden=solver_hidden,
            kernel=solver_kernel,
            share_across_iter=solver_share,
            dropout_p=solver_dropout,
            clip_each_step=clip_solver_steps,
            use_attention=solver_use_attention,
            attn_heads=solver_attn_heads,
            momentum_beta=solver_momentum,
            rnn_type=solver_rnn_type,
            lr_grad=solver_lr_grad,
            use_obs_encoder=solver_use_obs_encoder,
        )

        self.predict_uncertainty = bool(predict_uncertainty)
        self.log_var_min, self.log_var_max = float(log_var_clamp[0]), float(log_var_clamp[1])
        self.var_head: Optional[_VarHead] = None
        if self.predict_uncertainty:
            self.var_head = _VarHead(in_ch=4, mid=32, out_ch=3)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _solver_forward(
        self, history: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        # Ensure all inputs are pure torch.Tensor (not GridData subclass)
        if type(history) is not torch.Tensor:
            history = torch.empty(history.shape, dtype=history.dtype, device=history.device).copy_(history)
        if type(obs) is not torch.Tensor:
            obs = torch.empty(obs.shape, dtype=obs.dtype, device=obs.device).copy_(obs)
        if type(obs_mask) is not torch.Tensor:
            obs_mask = torch.empty(obs_mask.shape, dtype=obs_mask.dtype, device=obs_mask.device).copy_(obs_mask)

        x_prior = self.adapter(history)
        x_init = obs * obs_mask + x_prior * (1.0 - obs_mask)
        return self.solver(x_init, self.cost_fn, obs, obs_mask, x_prior)

    def _maybe_var(self, x_hat: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.predict_uncertainty or self.var_head is None:
            return None
        log_var = self.var_head(x_hat)
        return log_var.clamp(self.log_var_min, self.log_var_max)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def forward(
        self, history: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        """主前向：返回 ``x_hat`` ``[B, 4, H, W]``。

        即使 ``predict_uncertainty=True`` 也只返回 mean，方便 evaluation 代码
        无需修改。需要 log_var 时调用 ``forward_with_var``。
        """
        return self._solver_forward(history, obs, obs_mask)

    def forward_with_var(
        self, history: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回 ``(x_hat, log_var)``；当 ``predict_uncertainty=False`` 时 log_var 为 None。"""
        x_hat = self._solver_forward(history, obs, obs_mask)
        log_var = self._maybe_var(x_hat)
        return x_hat, log_var

    def compute_loss(
        self,
        history: torch.Tensor,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算训练损失和诊断指标（仅用于一次性 forward 验证场景，不含 NLLL）。"""
        # Ensure pure tensors
        if type(history) is not torch.Tensor:
            history = torch.empty(history.shape, dtype=history.dtype, device=history.device).copy_(history)
        if type(obs) is not torch.Tensor:
            obs = torch.empty(obs.shape, dtype=obs.dtype, device=obs.device).copy_(obs)
        if type(obs_mask) is not torch.Tensor:
            obs_mask = torch.empty(obs_mask.shape, dtype=obs_mask.dtype, device=obs_mask.device).copy_(obs_mask)
        if type(x_gt) is not torch.Tensor:
            x_gt = torch.empty(x_gt.shape, dtype=x_gt.dtype, device=x_gt.device).copy_(x_gt)

        x_prior = self.adapter(history)
        x_init = obs * obs_mask + x_prior * (1.0 - obs_mask)
        x_hat = self.solver(x_init, self.cost_fn, obs, obs_mask, x_prior)

        dm = density_support_mask(x_gt, self.cost_fn.rho_mask_thr)
        full = torch.ones_like(dm)
        m4 = torch.cat([full, dm, dm, dm], dim=1)
        recon_loss = masked_mean_sq(
            (x_hat - x_gt).pow(2) * self.cost_fn.ch_w, m4
        )

        with torch.no_grad():
            _, obs_t, prior_t = self.cost_fn(x_hat, obs, obs_mask, x_prior)
            phi_mse = masked_mean_sq((x_prior - x_gt).pow(2), m4)

        rho_m = dm
        info = {
            "recon": float(recon_loss.item()),
            "obs_term": float(obs_t.item()),
            "prior_term": float(prior_t.item()),
            "phi_mse": float(phi_mse.item()),
            "rho_mse": float(
                masked_mean_sq((x_hat[:, 0:1] - x_gt[:, 0:1]).pow(2), rho_m).item()
            ),
            "vx_mse": float(
                masked_mean_sq((x_hat[:, 1:2] - x_gt[:, 1:2]).pow(2), rho_m).item()
            ),
            "vy_mse": float(
                masked_mean_sq((x_hat[:, 2:3] - x_gt[:, 2:3]).pow(2), rho_m).item()
            ),
            "var_mse": float(
                masked_mean_sq((x_hat[:, 3:4] - x_gt[:, 3:4]).pow(2), rho_m).item()
            ),
            "density_mask_frac": float(dm.mean().item()),
        }
        return recon_loss, info
