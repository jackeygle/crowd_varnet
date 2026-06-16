"""变分代价项：观测项 + 基于 PedPred 先验的项，以及若干状态/掩码辅助函数。"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class VariationalCost(nn.Module):
    """同化代价：观测项（``obs_mask`` / 密度掩码）+ 相对 PedPred 先验 ``x_prior`` 的项。"""

    def __init__(
        self,
        w_obs: float = 1.0,
        w_prior: float = 0.5,
        ch_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0),
        rho_mask_thr: float = 0.05,
        prior_use_ch_weights: bool = True,
        prior_unobs_weight: float = 0.05,
    ):
        super().__init__()
        self.w_obs = w_obs
        self.w_prior = w_prior
        self.rho_mask_thr = rho_mask_thr
        self.prior_use_ch_weights = prior_use_ch_weights
        self.prior_unobs_weight = float(prior_unobs_weight)
        w = torch.tensor(ch_weights, dtype=torch.float32).view(1, 4, 1, 1)
        self.register_buffer("ch_w", w)

    def obs_term(self, x: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        err = (x - obs).pow(2)
        rho_mask = (obs[:, 0:1] > self.rho_mask_thr).float()
        # density: penalize everywhere sensor covers (not just where density > thr)
        # velocity: only where sensor sees pedestrians
        combined = torch.cat(
            [
                obs_mask,
                obs_mask * rho_mask,
                obs_mask * rho_mask,
                obs_mask * rho_mask,
            ],
            dim=1,
        )
        return (err * self.ch_w * combined).sum() / (combined.sum() + 1e-6)

    def prior_term(
        self, x: torch.Tensor, x_prior: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        err = (x - x_prior).pow(2)
        if self.prior_use_ch_weights:
            err = err * self.ch_w
        if obs_mask is not None:
            # Strong constraint in observed region (weight=1.0),
            # weak constraint in unobserved region (weight=prior_unobs_weight).
            # This gives the solver freedom to use attention/RNN to infer
            # unobserved regions instead of being pulled back to prior.
            weight = obs_mask + self.prior_unobs_weight * (1.0 - obs_mask)
            # Broadcast weight [B, 1, H, W] to all 4 channels
            weight4 = weight.expand(-1, 4, -1, -1)
            return (err * weight4).sum() / (weight4.sum() + 1e-6)
        else:
            # Fallback: full-image (no obs_mask provided)
            return err.mean()

    def forward(
        self,
        x: torch.Tensor,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        x_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o = self.obs_term(x, obs, obs_mask)
        p = self.prior_term(x, x_prior, obs_mask=obs_mask)
        return self.w_obs * o + self.w_prior * p, o, p


def clip_crowd_state(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    out[:, 0:1].clamp_(0.0, 5.0)
    out[:, 1:3].clamp_(-5.0, 5.0)
    out[:, 3:4].clamp_(0.0, 2.0)
    return out


def density_support_mask(x_gt: torch.Tensor, rho_thr: float) -> torch.Tensor:
    """Binary mask [B,1,H,W]: 1 where ground-truth density exceeds ``rho_thr``."""
    return (x_gt[:, 0:1] > rho_thr).to(dtype=x_gt.dtype)


def masked_mean_sq(err: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean of ``err`` over entries where ``mask`` is 1; ``mask`` broadcastable to ``err``."""
    m = mask
    if m.shape != err.shape:
        m = m.expand_as(err)
    den = m.sum().clamp_min(eps)
    return (err * m).sum() / den


def topk_mean_sq(
    err: torch.Tensor,
    mask: torch.Tensor,
    topk_percent: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean of the top-k% largest masked entries of ``err``.

    Same interface as ``masked_mean_sq`` but only the largest ``topk_percent``
    fraction of masked errors contributes to the loss. This is the OHEM /
    "hard example mining" trick adapted to a regression task — focuses gradient
    on the pixels the model is currently getting most wrong.

    Args:
        err: per-element error tensor (already squared and channel-weighted).
        mask: broadcastable mask; only positions with mask>0 are considered.
        topk_percent: fraction in (0, 1]; 1.0 falls back to standard masked mean.
        eps: numerical floor.

    Returns:
        Scalar mean over the top-k% selected entries.
    """
    if topk_percent >= 1.0 or topk_percent <= 0.0:
        return masked_mean_sq(err, mask, eps=eps)

    m = mask
    if m.shape != err.shape:
        m = m.expand_as(err)
    valid = m > 0
    if not valid.any():
        # Fall back: no valid entries
        return (err * m).sum() / (m.sum().clamp_min(eps))

    # Flatten valid errors and pick top-K
    valid_errs = err[valid]
    n = valid_errs.numel()
    k = max(1, int(n * topk_percent))
    topk_vals, _ = valid_errs.topk(k)
    return topk_vals.mean()
