"""Unified observation generator — same geometry for all methods.

Produces both:
  - C_joint (sparse observation matrix) for EnKF/PF
  - obs_mask [1, H, W] for CrowdVarNet
  - obs [4, H, W] = x_gt * obs_mask (noise-free partial observation)
  - obs_noisy [4, H, W] (with Gaussian noise, for EnKF)
"""
from __future__ import annotations

import sys
from typing import List, Tuple

import numpy as np

from .config import (
    GRID_SIZE, STATE_SHAPE, NUM_AGENTS, SENSING_RANGE,
    OBS_NOISE_STD, PO_ROOT, RANDOM_SEED,
)


def _get_multi_agent_class():
    """Import MultiAgent from Partial_observation."""
    po_parent = str(PO_ROOT.parent)
    if po_parent not in sys.path:
        sys.path.insert(0, po_parent)
    from Partial_observation.ENKF import MultiAgent
    return MultiAgent


def create_observation_sequence(
    frames: List[np.ndarray],
    num_agents: int = NUM_AGENTS,
    sensing_range: int = SENSING_RANGE,
    seed: int = RANDOM_SEED,
    add_noise: bool = True,
) -> List[dict]:
    """Generate observation sequence aligned with both EnKF and CrowdVarNet.

    Returns list of dicts, one per step:
        {
            "C_joint": np.ndarray,       # observation matrix for EnKF
            "obs_mask": np.ndarray,      # [1, H, W] binary mask for CrowdVarNet
            "obs_clean": np.ndarray,     # [4, H, W] noise-free partial obs
            "obs_noisy_vec": np.ndarray, # noisy observation vector for EnKF
            "observed_cells": list,      # list of (r, c) tuples
        }
    """
    MultiAgent = _get_multi_agent_class()

    # Use same RNG seed as Partial_observation's main()
    np.random.seed(seed)
    agents = MultiAgent(GRID_SIZE, STATE_SHAPE,
                        sensing_range=sensing_range, num_agents=num_agents)

    H, W = GRID_SIZE
    obs_noise_vec_full = np.concatenate([
        np.full(H * W, s) for s in OBS_NOISE_STD
    ])

    observations = []
    for step_idx, frame in enumerate(frames):
        # Move agents (same as EnKF main loop)
        agents.move_agents()

        # Joint observation matrix
        C_joint, observed_cells = agents.get_joint_observation()

        # True observation vector
        true_state_flat = frame.flatten()
        obs_vec_clean = C_joint @ true_state_flat

        # Noisy observation (for EnKF)
        obs_idx = C_joint.nonzero()[1]
        if add_noise:
            noise = np.random.normal(0, obs_noise_vec_full[obs_idx])
            obs_vec_noisy = obs_vec_clean + noise
        else:
            obs_vec_noisy = obs_vec_clean.copy()

        # Build obs_mask [1, H, W] for CrowdVarNet
        obs_mask = np.zeros((1, H, W), dtype=np.float32)
        for r, c in observed_cells:
            obs_mask[0, r, c] = 1.0

        # Build obs [4, H, W] = frame * obs_mask (noise-free for CrowdVarNet)
        obs_clean = frame * obs_mask

        observations.append({
            "C_joint": C_joint,
            "obs_mask": obs_mask,
            "obs_clean": obs_clean,
            "obs_noisy_vec": obs_vec_noisy,
            "observed_cells": observed_cells,
        })

    return observations
