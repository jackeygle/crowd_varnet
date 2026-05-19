"""CrowdVarNet: variational crowd-state reconstruction with frozen PedPred prior."""

from .datasets import (
    CrowdVarNetDataset,
    RolloutEpisodeDataset,
    spatial_sensor_mask,
    stack_grid_sequence,
)
from .models import (
    CrowdVarNet,
    CrowdVarNetIterativeSolver,
    FrozenPedPredPrior,
    VariationalCost,
    clip_crowd_state,
    load_frozen_pedpred,
)
from .training import rollout_tbptt_epoch, rollout_val_loss

# 旧名（向后兼容）
from .assimilation_model import LearnedGradSolver, PedPredAdapter

__all__ = [
    "CrowdVarNet",
    "CrowdVarNetDataset",
    "CrowdVarNetIterativeSolver",
    "FrozenPedPredPrior",
    "RolloutEpisodeDataset",
    "VariationalCost",
    "clip_crowd_state",
    "load_frozen_pedpred",
    "rollout_tbptt_epoch",
    "rollout_val_loss",
    "spatial_sensor_mask",
    "stack_grid_sequence",
    # 旧名
    "LearnedGradSolver",
    "PedPredAdapter",
]
