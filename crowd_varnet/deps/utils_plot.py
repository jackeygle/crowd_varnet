"""Plotting and step NPZ helpers (vendored from partial_observation_experiments.utils)."""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

STATE_SHAPE_DEFAULT = (4, 36, 12)


def plot_generated_matrix_on_ax(example_matrix, ax, state_shape=(4, 36, 12)):
    if example_matrix.ndim == 1:
        if state_shape is None:
            raise ValueError("state_shape must be provided for flattened input")
        example_matrix = example_matrix.reshape(state_shape)
    density = example_matrix[0]
    velocity_x_matrix = example_matrix[1]
    velocity_y_matrix = example_matrix[2]
    velocity = np.sqrt(velocity_x_matrix**2 + velocity_y_matrix**2)
    heading = np.arctan2(velocity_y_matrix, velocity_x_matrix)

    x, y = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
    u = np.cos(heading)
    v = np.sin(heading)

    mask = np.isnan(density) | (heading == 0)
    u[mask] = np.nan
    v[mask] = np.nan

    im = ax.imshow(density, cmap="Blues", aspect="auto", origin="upper", vmin=0, vmax=1)
    ax.quiver(x, y, u, v, color="black", scale=30, headwidth=3, headlength=4)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def dump_step_arrays_npz(
    run_dir: str,
    step_idx: int,
    partial_obs,
    true_state,
    estimated_mean,
    estimated_spread,
    *,
    state_shape: Tuple[int, int, int] = STATE_SHAPE_DEFAULT,
) -> str:
    out_dir = os.path.join(run_dir, "step_npz")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"step_{step_idx:03d}.npz")

    def f32_row(x):
        a = np.asarray(x, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(state_shape)
        return a

    np.savez_compressed(
        path,
        partial_obs=f32_row(partial_obs),
        true_state=f32_row(true_state),
        estimated_mean=f32_row(estimated_mean),
        estimated_spread=f32_row(estimated_spread),
        step_idx=np.int32(step_idx),
        state_shape=np.array(state_shape, dtype=np.int32),
    )
    return path


def load_step_arrays_npz(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {
        "partial_obs": np.asarray(z["partial_obs"]),
        "true_state": np.asarray(z["true_state"]),
        "estimated_mean": np.asarray(z["estimated_mean"]),
        "estimated_spread": np.asarray(z["estimated_spread"]),
        "step_idx": int(z["step_idx"]),
        "state_shape": tuple(np.asarray(z["state_shape"]).tolist()),
    }


def save_step_plot(
    partial_obs,
    true_state,
    estimated_mean,
    estimated_spread,
    step_idx,
    run_dir,
    *,
    spread_vmax=None,
):
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.15)

    ax0 = fig.add_subplot(gs[0])
    im0 = plot_generated_matrix_on_ax(partial_obs, ax0)
    ax0.set_title(f"Step {step_idx} - Partial Obs")

    ax1 = fig.add_subplot(gs[1])
    im1 = plot_generated_matrix_on_ax(true_state, ax1)
    ax1.set_title("True State")

    ax2 = fig.add_subplot(gs[2])
    im2 = plot_generated_matrix_on_ax(estimated_mean, ax2)
    ax2.set_title("Estimated Mean")

    ax3 = fig.add_subplot(gs[3])
    density_spread = estimated_spread.copy()
    density_spread[np.isnan(density_spread)] = 0
    rho_sigma = density_spread[0]
    smean = float(np.mean(rho_sigma))
    slocmax = float(np.max(rho_sigma))
    vmax_note = ""
    if spread_vmax is not None and float(spread_vmax) > 0:
        vmax_note = f" | color vmax={float(spread_vmax):.4f}"
        im3 = ax3.imshow(
            rho_sigma,
            cmap="Reds",
            origin="upper",
            aspect="auto",
            vmin=0.0,
            vmax=float(spread_vmax),
        )
    else:
        im3 = ax3.imshow(rho_sigma, cmap="Reds", origin="upper", aspect="auto", vmin=0.0)
    ax3.set_title(f"Estimated Spread (σρ)\nmean={smean:.4f}  max={slocmax:.4f}{vmax_note}")
    ax3.set_xticks([])
    ax3.set_yticks([])

    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    filename = os.path.join(run_dir, f"step_{step_idx:03d}.png")
    plt.savefig(filename)
    plt.close(fig)
