"""端到端 CrowdVarNet：先验 + 初值融合 + 迭代求解；``compute_loss`` 为训练目标。"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .cost import VariationalCost, density_support_mask, masked_mean_sq
from .prior import FrozenPedPredPrior
from .solver import CrowdVarNetIterativeSolver


class InitGate(nn.Module):
    """可学习的初值融合门：根据 ``[x_prior, obs, obs_mask]`` 估计 per-pixel/per-channel α∈[0,1]。

    ``x_init = α * obs + (1 - α) * x_prior``。
    末层零初始化 + bias=logit(mask)：开局等价于硬切换 ``mask*obs + (1-mask)*x_prior``，向后兼容。
    """

    def __init__(self, mid: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4 + 4 + 1, mid, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, mid), num_channels=mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 4, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self, x_prior: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        # 让 logit 的偏置由 mask 决定：mask=1 → α≈1（用 obs），mask=0 → α≈0（用 prior）。
        feat = torch.cat([x_prior, obs, obs_mask], dim=1)
        delta_logit = self.body(feat)
        # 把 mask 复制到 4 通道作为 base logit；scale 让 sigmoid 接近硬切换。
        base = (obs_mask * 2.0 - 1.0) * 6.0  # mask=1→+6, mask=0→-6 → sigmoid≈{1,0}
        alpha = torch.sigmoid(base + delta_logit)
        return alpha * obs + (1.0 - alpha) * x_prior


class CrowdVarNet(nn.Module):
    """
    训练与验证的 ``recon``、分量 MSE、``phi_mse`` 均在 ``x_gt`` 密度 > ``rho_mask_thr`` 的格点上平均。
    ``forward`` 只依赖 (history, obs, obs_mask)；``compute_loss`` 外层用 ``x_gt`` 建加权 mask（不进 solver）。
    """

    def __init__(
        self,
        ped_pred: nn.Module,
        freeze_phi: bool = True,
        T_hist: int = 5,
        n_iter: int = 8,
        use_gru: bool = False,
        gru_ch: int = 16,
        w_obs: float = 1.0,
        w_prior: float = 0.5,
        ch_weights: Tuple[float, float, float, float] = (2.5, 1.5, 1.0, 0.5),
        rho_mask_thr: float = 0.05,
        prior_use_ch_weights: bool = True,
        clip_solver_steps: bool = True,
        *,
        solver_type: Optional[str] = None,
        solver_hidden: int = 32,
        solver_kernel: int = 3,
        solver_share: bool = True,
        solver_dropout: float = 0.0,
        init_gate: bool = False,
        init_gate_mid: int = 16,
        unfreeze_phi_tail: int = 0,
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
        )
        self.solver = CrowdVarNetIterativeSolver(
            n_iter,
            use_gru=use_gru,
            gru_ch=gru_ch,
            clip_each_step=clip_solver_steps,
            solver_type=solver_type,
            hidden=solver_hidden,
            kernel=solver_kernel,
            share_across_iter=solver_share,
            dropout_p=solver_dropout,
        )
        self.init_gate = InitGate(mid=init_gate_mid) if init_gate else None

    def _compute_x_init(
        self, x_prior: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.init_gate is None:
            return obs * obs_mask + x_prior * (1.0 - obs_mask)
        return self.init_gate(x_prior, obs, obs_mask)

    def forward(
        self, history: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> torch.Tensor:
        x_prior = self.adapter(history)
        x_init = self._compute_x_init(x_prior, obs, obs_mask)
        return self.solver(x_init, self.cost_fn, obs, obs_mask, x_prior)

    def compute_loss(
        self,
        history: torch.Tensor,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # compute x_prior once and reuse — avoids a second adapter forward pass
        x_prior = self.adapter(history)
        x_init = self._compute_x_init(x_prior, obs, obs_mask)
        x_hat = self.solver(x_init, self.cost_fn, obs, obs_mask, x_prior)

        dm = density_support_mask(x_gt, self.cost_fn.rho_mask_thr)
        # density channel: penalize everywhere (full mask); velocity channels: only where GT has pedestrians
        full = torch.ones_like(dm)
        m4 = torch.cat([full, dm, dm, dm], dim=1)
        recon_loss = masked_mean_sq(
            (x_hat - x_gt).pow(2) * self.cost_fn.ch_w, m4
        )

        # auxiliary loss so InitGate receives gradients (solver detaches x_init internally)
        if self.init_gate is not None:
            init_loss = masked_mean_sq((x_init - x_gt).pow(2) * self.cost_fn.ch_w, m4)
            recon_loss = recon_loss + 0.1 * init_loss

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
