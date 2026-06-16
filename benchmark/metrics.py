"""Unified evaluation metrics for all methods."""
from __future__ import annotations

from typing import Dict

import numpy as np


def per_channel_rmse(x_hat: np.ndarray, x_gt: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """Per-channel RMSE. x_hat, x_gt: [4, H, W]. mask: [H, W] or None.
    Only evaluates first 3 channels (rho, vx, vy). Ignores variance channel."""
    from .config import CH_NAMES
    out = {}
    for c, name in enumerate(CH_NAMES[:3]):  # only rho, vx, vy
        diff = x_hat[c] - x_gt[c]
        if mask is not None:
            diff = diff[mask > 0.5]
        if diff.size == 0:
            out[name] = 0.0
        else:
            out[name] = float(np.sqrt(np.mean(diff ** 2)))
    return out


def per_channel_mae(x_hat: np.ndarray, x_gt: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """Per-channel MAE. x_hat, x_gt: [4, H, W]. mask: [H, W] or None.
    Only evaluates first 3 channels (rho, vx, vy)."""
    from .config import CH_NAMES
    out = {}
    for c, name in enumerate(CH_NAMES[:3]):  # only rho, vx, vy
        diff = np.abs(x_hat[c] - x_gt[c])
        if mask is not None:
            diff = diff[mask > 0.5]
        if diff.size == 0:
            out[name] = 0.0
        else:
            out[name] = float(np.mean(diff))
    return out


def full_state_rmse(x_hat: np.ndarray, x_gt: np.ndarray, rho_thr: float = 0.05) -> float:
    """Full state RMSE on support pixels (where GT density > thr).
    Only first 3 channels (rho, vx, vy)."""
    mask = x_gt[0] > rho_thr  # [H, W]
    if mask.sum() == 0:
        return 0.0
    err = 0.0
    for c in range(3):
        err += np.mean((x_hat[c][mask] - x_gt[c][mask]) ** 2)
    return float(np.sqrt(err / 3.0))


def density_support_mask(x_gt: np.ndarray, rho_thr: float = 0.05) -> np.ndarray:
    """Binary mask [H, W]: 1 where GT density > rho_thr."""
    return (x_gt[0] > rho_thr).astype(np.float32)


def density_mae(x_hat: np.ndarray, x_gt: np.ndarray) -> float:
    """Density channel MAE (compatible with Partial_observation)."""
    return float(np.mean(np.abs(x_hat[0] - x_gt[0])))


def velocity_weighted_mae(x_hat: np.ndarray, x_gt: np.ndarray) -> float:
    """Density-weighted velocity MAE (compatible with Partial_observation)."""
    rho_gt = x_gt[0]  # [H, W]
    vel_diff = x_hat[1:3] - x_gt[1:3]  # [2, H, W]
    weighted = rho_gt * np.sqrt(vel_diff[0] ** 2 + vel_diff[1] ** 2)
    return float(np.mean(weighted))


def compute_all_metrics(
    x_hat: np.ndarray, x_gt: np.ndarray, rho_thr: float = 0.05
) -> Dict[str, float]:
    """Compute all metrics for a single frame. Returns flat dict."""
    mask = density_support_mask(x_gt, rho_thr)
    out = {}
    out["full_rmse"] = full_state_rmse(x_hat, x_gt)
    out["density_mae"] = density_mae(x_hat, x_gt)
    out["velocity_weighted_mae"] = velocity_weighted_mae(x_hat, x_gt)

    # Per-channel RMSE (global)
    ch_rmse_global = per_channel_rmse(x_hat, x_gt, mask=None)
    for k, v in ch_rmse_global.items():
        out[f"rmse_global_{k}"] = v

    # Per-channel RMSE (support only)
    ch_rmse_support = per_channel_rmse(x_hat, x_gt, mask=mask)
    for k, v in ch_rmse_support.items():
        out[f"rmse_support_{k}"] = v

    # Per-channel MAE (support only)
    ch_mae_support = per_channel_mae(x_hat, x_gt, mask=mask)
    for k, v in ch_mae_support.items():
        out[f"mae_support_{k}"] = v

    out["density_mask_frac"] = float(mask.mean())
    return out
