"""Unified method wrappers — each method exposes the same interface.

Each method class implements:
    initialize(first_frame: np.ndarray) -> None
    step(obs_dict: dict, true_frame: np.ndarray) -> np.ndarray  # returns x_hat [4, H, W]
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np
import torch

from .config import (
    GRID_SIZE, STATE_SHAPE, STATE_DIM, NUM_CHANNELS,
    ENKF_ENSEMBLE_SIZE, ENKF_PROC_STD, ENKF_OBS_STD,
    ENKF_LOC_RADIUS, ENKF_INFLATION,
    OUR_TEACHER_CKPT, OUR_TEACHER_ARCH,
    THEIR_TEACHER_CKPT, CVN_BEST_CKPT,
    PO_ROOT, WARMUP_STEPS,
)


class BaseMethod:
    """Base class for all methods."""
    name: str = "base"

    def initialize(self, first_frame: np.ndarray) -> None:
        raise NotImplementedError

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        """Returns estimated state [4, H, W]."""
        raise NotImplementedError


# ============================================================
# EnKF (Localized, from Partial_observation)
# ============================================================
class EnKFMethod(BaseMethod):
    name = "EnKF (Localized)"

    def __init__(self, device: str = "cuda"):
        po_parent = str(PO_ROOT.parent)
        if po_parent not in sys.path:
            sys.path.insert(0, po_parent)
        from Partial_observation.ENKF import LocalizedEnsembleKalmanFilter
        from Partial_observation.utils import load_model

        self.device = device
        self.model = load_model(str(THEIR_TEACHER_CKPT), device)
        # Patch the model to CPU for EnKF (ensemble runs on CPU numpy)
        self.model = self.model.cpu()
        self.model.eval()
        self.enkf = LocalizedEnsembleKalmanFilter(
            grid_size=GRID_SIZE,
            state_shape=STATE_SHAPE,
            ensemble_size=ENKF_ENSEMBLE_SIZE,
            proc_noise_std=ENKF_PROC_STD,
            obs_noise_std=ENKF_OBS_STD,
            inflation=ENKF_INFLATION,
            localization_radius=ENKF_LOC_RADIUS,
        )

    def initialize(self, first_frame: np.ndarray) -> None:
        self.enkf.initialize(first_frame)

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        C_joint = obs_dict["C_joint"]
        obs_noisy_vec = obs_dict["obs_noisy_vec"]
        x_hat = self.enkf.step(C_joint, obs_noisy_vec, model=self.model)
        return x_hat


# ============================================================
# Particle Filter (from Partial_observation)
# ============================================================
class ParticleFilterMethod(BaseMethod):
    name = "Particle Filter"

    def __init__(self, device: str = "cuda", num_particles: int = 200):
        po_parent = str(PO_ROOT.parent)
        if po_parent not in sys.path:
            sys.path.insert(0, po_parent)
        from Partial_observation.PF import ParticleFilter
        from Partial_observation.utils import load_model

        self.device = device
        self.model = load_model(str(THEIR_TEACHER_CKPT), device)
        # PF ensemble runs on CPU numpy, model needs to be on CPU too
        self.model = self.model.cpu()
        self.model.eval()
        self.pf = ParticleFilter(
            grid_size=GRID_SIZE,
            state_shape=STATE_SHAPE,
            num_particles=num_particles,
            proc_noise_std=ENKF_PROC_STD,
            obs_noise_std=ENKF_OBS_STD,
        )

    def initialize(self, first_frame: np.ndarray) -> None:
        self.pf.initialize(first_frame)

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        C_joint = obs_dict["C_joint"]
        obs_noisy_vec = obs_dict["obs_noisy_vec"]
        self.pf.forecast(self.model)
        self.pf.update(C_joint, obs_noisy_vec)
        return self.pf.estimate()


# ============================================================
# CrowdVarNet (ours)
# ============================================================
class CrowdVarNetMethod(BaseMethod):
    name = "CrowdVarNet (ours)"

    def __init__(self, device: str = "cuda"):
        import os
        cvn_root = str(CVN_BEST_CKPT.parent.parent.parent)
        if cvn_root not in sys.path:
            sys.path.insert(0, cvn_root)

        from crowd_varnet.cli import build_model_from_ckpt

        # Allow env var override for testing different student checkpoints
        ckpt_path = os.environ.get("CVN_BEST_CKPT", str(CVN_BEST_CKPT))
        from pathlib import Path
        ckpt_path = Path(ckpt_path)

        self.device = torch.device(device)
        self.model, self.meta = build_model_from_ckpt(
            ckpt_path, device=self.device
        )
        self.model.eval()
        self.T_hist = int(self.model.T_hist)
        self.history_buf = None

    def initialize(self, first_frame: np.ndarray) -> None:
        # Fill history buffer with the first frame repeated T_hist times
        frame_t = torch.from_numpy(first_frame).float().unsqueeze(0)  # [1, 4, H, W]
        self.history_buf = frame_t.repeat(1, self.T_hist, 1, 1, 1).to(self.device)
        # [1, T_hist, 4, H, W]

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        obs_mask = torch.from_numpy(obs_dict["obs_mask"]).float().unsqueeze(0).to(self.device)
        obs = torch.from_numpy(obs_dict["obs_clean"]).float().unsqueeze(0).to(self.device)

        # Solver uses autograd internally, so we need grad enabled
        # but we don't want param gradients
        saved = [(p, p.requires_grad) for p in self.model.parameters() if p.requires_grad]
        for p, _ in saved:
            p.requires_grad_(False)
        try:
            with torch.set_grad_enabled(True):
                x_hat = self.model.forward(self.history_buf, obs, obs_mask)
        finally:
            for p, was in saved:
                p.requires_grad_(was)

        x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)

        # Update history buffer (autoregressive)
        self.history_buf = torch.cat(
            [self.history_buf[:, 1:], x_hat.detach().unsqueeze(1)], dim=1
        )
        return x_hat_np


# ============================================================
# Naive baseline: obs * mask + PedPred(history) * (1 - mask)
# ============================================================
class NaiveMethod(BaseMethod):
    name = "Naive (obs + prior)"

    def __init__(self, device: str = "cuda"):
        cvn_root = str(CVN_BEST_CKPT.parent.parent.parent)
        if cvn_root not in sys.path:
            sys.path.insert(0, cvn_root)

        from crowd_varnet.models import load_frozen_pedpred
        from crowd_varnet.models.prior import FrozenPedPredPrior

        self.device = torch.device(device)
        ped = load_frozen_pedpred(str(OUR_TEACHER_CKPT), self.device, arch=OUR_TEACHER_ARCH)
        self.prior = FrozenPedPredPrior(ped, freeze=True).to(self.device).eval()
        self.T_hist = WARMUP_STEPS
        self.history_buf = None

    def initialize(self, first_frame: np.ndarray) -> None:
        frame_t = torch.from_numpy(first_frame).float().unsqueeze(0)
        self.history_buf = frame_t.repeat(1, self.T_hist, 1, 1, 1).to(self.device)

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        obs_mask = torch.from_numpy(obs_dict["obs_mask"]).float().unsqueeze(0).to(self.device)
        obs = torch.from_numpy(obs_dict["obs_clean"]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            x_prior = self.prior(self.history_buf)
        x_hat = obs * obs_mask + x_prior * (1.0 - obs_mask)
        x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)

        self.history_buf = torch.cat(
            [self.history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1
        )
        return x_hat_np


# ============================================================
# PedPred-only: pure open-loop forecast, ignores observations
# ============================================================
class PedPredOnlyMethod(BaseMethod):
    name = "PedPred-only (no obs)"

    def __init__(self, device: str = "cuda"):
        cvn_root = str(CVN_BEST_CKPT.parent.parent.parent)
        if cvn_root not in sys.path:
            sys.path.insert(0, cvn_root)

        from crowd_varnet.models import load_frozen_pedpred
        from crowd_varnet.models.prior import FrozenPedPredPrior

        self.device = torch.device(device)
        ped = load_frozen_pedpred(str(OUR_TEACHER_CKPT), self.device, arch=OUR_TEACHER_ARCH)
        self.prior = FrozenPedPredPrior(ped, freeze=True).to(self.device).eval()
        self.T_hist = WARMUP_STEPS
        self.history_buf = None

    def initialize(self, first_frame: np.ndarray) -> None:
        frame_t = torch.from_numpy(first_frame).float().unsqueeze(0)
        self.history_buf = frame_t.repeat(1, self.T_hist, 1, 1, 1).to(self.device)

    def step(self, obs_dict: dict, true_frame: np.ndarray = None) -> np.ndarray:
        with torch.no_grad():
            x_hat = self.prior(self.history_buf)
        x_hat_np = x_hat[0].detach().cpu().numpy().astype(np.float64)

        self.history_buf = torch.cat(
            [self.history_buf[:, 1:], x_hat.unsqueeze(1)], dim=1
        )
        return x_hat_np
