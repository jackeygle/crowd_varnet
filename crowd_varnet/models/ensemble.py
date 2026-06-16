"""Deep Ensemble of CrowdVarNet students.

Loads N independently-trained checkpoints (different ``--seed``) and provides
a single ``forward`` returning ensemble mean prediction and decomposed
aleatoric / epistemic / total uncertainty.

Convention:
  - Each member predicts (x_hat [B,4,H,W], log_var [B,3,H,W])  for ρ, vx, vy.
  - Aleatoric variance is the average of per-member exp(log_var).
  - Epistemic variance is the variance across members of x_hat[:, :3].
  - Total variance = aleatoric + epistemic.
  - Mean state x̄ keeps 4 channels; the 4th channel (legacy ``speed`` /
    teacher's σ²) is just averaged across members.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .varnet import CrowdVarNet


class CrowdVarNetEnsemble(nn.Module):
    """Inference-only wrapper around N CrowdVarNet members."""

    def __init__(self, members: Sequence[CrowdVarNet]):
        super().__init__()
        if len(members) == 0:
            raise ValueError("CrowdVarNetEnsemble requires at least one member")
        self.members = nn.ModuleList(list(members))
        if not all(getattr(m, "predict_uncertainty", False) for m in self.members):
            raise ValueError(
                "All ensemble members must have predict_uncertainty=True; "
                "otherwise aleatoric uncertainty is undefined."
            )

    @classmethod
    def from_checkpoints(
        cls,
        ckpt_paths: Sequence[Union[str, Path]],
        *,
        device: torch.device,
    ) -> "CrowdVarNetEnsemble":
        # Local import avoids circular dependency at package init time
        # (cli._common imports from models package).
        from ..cli._common import build_model_from_ckpt
        members: List[CrowdVarNet] = []
        for p in ckpt_paths:
            model, _meta = build_model_from_ckpt(p, device=device)
            model.eval()
            members.append(model)
        return cls(members)

    @torch.no_grad()
    def forward(
        self, history: torch.Tensor, obs: torch.Tensor, obs_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Run all members and return ensemble statistics.

        Returns dict with:
          - ``mean``: [B, 4, H, W] ensemble mean state
          - ``aleatoric``: [B, 3, H, W] mean of per-member σ² (data noise)
          - ``epistemic``: [B, 3, H, W] inter-member variance of mean (model)
          - ``total``: [B, 3, H, W] aleatoric + epistemic
          - ``members_mean``: [N, B, 4, H, W] per-member predictions
          - ``members_log_var``: [N, B, 3, H, W] per-member log-variances
        """
        means: List[torch.Tensor] = []
        log_vars: List[torch.Tensor] = []
        for m in self.members:
            x_hat, log_var = m.forward_with_var(history, obs, obs_mask)
            means.append(x_hat)
            assert log_var is not None, "all members must predict log_var"
            log_vars.append(log_var)
        means_t = torch.stack(means, dim=0)        # [N, B, 4, H, W]
        log_var_t = torch.stack(log_vars, dim=0)   # [N, B, 3, H, W]

        ens_mean = means_t.mean(dim=0)             # [B, 4, H, W]
        # Aleatoric: average of σ² across members (each member's data-noise estimate)
        aleatoric = torch.exp(log_var_t).mean(dim=0)            # [B, 3, H, W]
        # Epistemic: variance of mean predictions (first 3 channels) across members
        epistemic = means_t[:, :, :3].var(dim=0, unbiased=False)  # [B, 3, H, W]
        total = aleatoric + epistemic
        return {
            "mean": ens_mean,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total,
            "members_mean": means_t,
            "members_log_var": log_var_t,
        }
