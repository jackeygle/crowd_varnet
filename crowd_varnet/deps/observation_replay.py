"""
Export / load observation replay bundles for Teacher vs CrowdVarNet comparisons.
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Optional

import numpy as np
import torch

from .dataset_atc import get_atc_data
from .enkf_sensors import MultiAgent
from .grid_data import GridData

REPLAY_VERSION = 1


def _as_eval_tensor(x_batch: torch.Tensor) -> torch.Tensor:
    if isinstance(x_batch, torch.Tensor) and x_batch.ndim >= 3 and x_batch.shape[-3] == 4:
        return x_batch
    return GridData(x_batch).as_tensor("density", "vel_mean", "vel_var")


def collect_eval_frames(
    num_steps: int,
    *,
    period: float = 1.0,
    batch: Optional[int] = None,
    nin: int = 1,
    nout: int = 1,
    num_workers: int = 0,
):
    data = get_atc_data(
        "test",
        period=period,
        batch=batch if batch is not None else int(os.environ.get("PEDPRED_BATCH", "16")),
        nin=nin,
        nout=nout,
        num_workers=num_workers,
        drop_last=False,
    )
    frames = []
    need = int(num_steps)
    for x_batch, _ in data:
        xb = _as_eval_tensor(x_batch)
        if xb.ndim != 5:
            raise ValueError(f"expected [B,T,C,H,W], got {tuple(xb.shape)}")
        if xb.shape[1] != 1:
            raise RuntimeError(
                "Align runners: export PEDPRED_NIN=1 PEDPRED_NOUT=1 "
                f"(got T_in={xb.shape[1]})."
            )
        for bi in range(xb.shape[0]):
            if len(frames) >= need:
                break
            frames.append(xb[bi, 0])
        if len(frames) >= need:
            break
    if len(frames) < need:
        raise RuntimeError(
            f"Not enough test samples: {len(frames)} < {need}. "
            "Increase PEDPRED_BATCH or check data."
        )
    return frames


def build_replay_bundle(
    *,
    num_steps: int,
    grid_size: tuple[int, int],
    state_shape: tuple[int, int, int],
    sensing_range: int,
    num_agents: int,
    obs_std: np.ndarray,
    replay_agent_seed: int,
    replay_obs_seed: int,
    data_period: float = 1.0,
    data_batch: Optional[int] = None,
) -> dict[str, Any]:
    frames = collect_eval_frames(
        num_steps,
        period=data_period,
        batch=data_batch,
        nin=1,
        nout=1,
    )
    rng_a = np.random.RandomState(int(replay_agent_seed))
    rng_o = np.random.RandomState(int(replay_obs_seed))
    agents = MultiAgent(
        grid_size,
        state_shape,
        sensing_range=sensing_range,
        num_agents=num_agents,
        rng=rng_a,
    )
    obs_std = np.asarray(obs_std, dtype=np.float64).reshape(4,)
    obs_noise_expand = np.concatenate([np.full(grid_size[0] * grid_size[1], s) for s in obs_std])
    entries = []
    for step_idx in range(num_steps):
        agents.move_agents()
        C_joint, _ = agents.get_joint_observation()
        true_state = frames[step_idx].detach().cpu().numpy().astype(np.float64).flatten()
        y_clean = C_joint @ true_state
        obs_idx = C_joint.nonzero()[1]
        y_obs = y_clean + rng_o.normal(0, obs_noise_expand[obs_idx], size=y_clean.shape)
        entries.append(
            {
                "C_joint": np.asarray(C_joint, dtype=np.float64),
                "y_clean": np.asarray(y_clean, dtype=np.float64),
                "y_obs": np.asarray(y_obs, dtype=np.float32),
            }
        )
    return {
        "version": REPLAY_VERSION,
        "obs_std": obs_std,
        "GRID_SIZE": tuple(grid_size),
        "STATE_SHAPE": tuple(state_shape),
        "sensing_range": int(sensing_range),
        "num_agents": int(num_agents),
        "replay_agent_seed": int(replay_agent_seed),
        "replay_obs_seed": int(replay_obs_seed),
        "entries": entries,
    }


def save_replay_bundle(path: str, bundle: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_replay_bundle(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if int(bundle.get("version", -1)) != REPLAY_VERSION:
        raise ValueError(f"unsupported replay version {bundle.get('version')!r}; need {REPLAY_VERSION}")
    return bundle


def export_cli():
    p = argparse.ArgumentParser(description="Export teacher-style MultiAgent observation replay (C, y).")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--data-period", type=float, default=1.0)
    p.add_argument("--data-batch", type=int, default=None, help="Test loader batch (default: PEDPRED_BATCH)")
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--sensing-range", type=int, default=5)
    p.add_argument("--replay-agent-seed", type=int, default=0)
    p.add_argument("--replay-obs-seed", type=int, default=1)
    p.add_argument(
        "--obs-std",
        type=float,
        nargs=4,
        metavar=("RHO", "VX", "VY", "VAR"),
        default=[0.05690936, 0.31472941, 0.08616199, 0.00644609],
    )
    args = p.parse_args()
    grid_size = (36, 12)
    state_shape = (4, 36, 12)
    bundle = build_replay_bundle(
        num_steps=int(args.num_steps),
        grid_size=grid_size,
        state_shape=state_shape,
        sensing_range=int(args.sensing_range),
        num_agents=int(args.num_agents),
        obs_std=np.array(args.obs_std, dtype=np.float64),
        replay_agent_seed=int(args.replay_agent_seed),
        replay_obs_seed=int(args.replay_obs_seed),
        data_period=float(args.data_period),
        data_batch=args.data_batch,
    )
    save_replay_bundle(args.out, bundle)
    print(
        f"[observation_replay] wrote {args.out} | steps={len(bundle['entries'])} | "
        f"agent_seed={args.replay_agent_seed} obs_seed={args.replay_obs_seed}",
        flush=True,
    )


if __name__ == "__main__":
    export_cli()
