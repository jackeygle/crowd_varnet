"""Unified data loader — loads the same test sequence for all methods.

Uses our own dataset_atc pipeline (which reads from grid_cache directly)
to avoid path issues with Partial_observation's relative paths.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .config import DATA_ROOT, GRID_SIZE, STATE_SHAPE, PO_ROOT


def get_test_sequence(split: str = "test", max_steps: int = 300) -> List[np.ndarray]:
    """Load ATC test data as a list of numpy arrays [4, H, W].

    Uses crowd_varnet's own data pipeline (reads grid_cache H5 directly).
    """
    # Use our own data loader which handles paths correctly
    cvn_root = str(Path(__file__).parent.parent)
    if cvn_root not in sys.path:
        sys.path.insert(0, cvn_root)

    import os
    os.environ.setdefault("PEDPRED_ATC_DATA_DIR", str(DATA_ROOT))
    os.environ.setdefault("PEDPRED_RESOLUTION", "1.0")
    os.environ.setdefault("PEDPRED_PERIOD", "1.0")
    os.environ.setdefault("PEDPRED_KERNEL", "tri")

    from crowd_varnet.deps.dataset_atc import get_atc_data
    from crowd_varnet.deps.grid_data import GridData

    loader = get_atc_data(
        split,
        batch=1,
        nin=1,
        nout=1,
        num_workers=0,
        drop_last=False,
    )

    frames = []
    for step_idx, (x_in, _) in enumerate(loader):
        if step_idx >= max_steps:
            break
        # x_in could be GridData or tensor; normalize to [4, H, W] numpy
        if hasattr(x_in, 'as_tensor'):
            t = GridData(x_in).as_tensor('density', 'vel_mean', 'vel_var')
        else:
            t = x_in
        if t.dim() == 5:
            t = t[0, 0]  # [B, T, C, H, W] -> [C, H, W]
        elif t.dim() == 4:
            t = t[0]  # [B, C, H, W] -> [C, H, W]
        frames.append(t.numpy().astype(np.float64))

    return frames


def get_test_sequence_with_history(
    split: str = "test", max_steps: int = 300, nin: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Load (history[T,4,H,W], target[4,H,W]) pairs for teacher evaluation.

    Each pair: history = nin consecutive frames, target = next frame.
    """
    cvn_root = str(Path(__file__).parent.parent)
    if cvn_root not in sys.path:
        sys.path.insert(0, cvn_root)

    import os
    os.environ.setdefault("PEDPRED_ATC_DATA_DIR", str(DATA_ROOT))
    os.environ.setdefault("PEDPRED_RESOLUTION", "1.0")
    os.environ.setdefault("PEDPRED_PERIOD", "1.0")
    os.environ.setdefault("PEDPRED_KERNEL", "tri")

    from crowd_varnet.deps.dataset_atc import get_atc_data
    from crowd_varnet.deps.grid_data import GridData
    from crowd_varnet.datasets.sensors import stack_grid_sequence

    loader = get_atc_data(
        split,
        batch=1,
        nin=nin,
        nout=1,
        num_workers=0,
        drop_last=False,
    )

    pairs = []
    for step_idx, (x_in, x_target) in enumerate(loader):
        if step_idx >= max_steps:
            break
        # history: [B, T, C, H, W] -> [T, C, H, W]
        hist = stack_grid_sequence(x_in[0] if hasattr(x_in, '__getitem__') else [x_in])
        if hist.dim() == 3:
            hist = hist.unsqueeze(0)
        # target: [B, 1, C, H, W] -> [C, H, W]
        if hasattr(x_target, 'as_tensor'):
            tgt = GridData(x_target).as_tensor('density', 'vel_mean', 'vel_var')
        else:
            tgt = x_target
        if tgt.dim() == 5:
            tgt = tgt[0, 0]
        elif tgt.dim() == 4:
            tgt = tgt[0]

        pairs.append((hist.numpy().astype(np.float64), tgt.numpy().astype(np.float64)))

    return pairs
