"""NLLL / weighted metrics for PedPred teacher training (uses package ``GridData``)."""
from __future__ import annotations

import torch

from .grid_data import GridData


class Metrics(dict):
    def __class_getitem__(cls, item):
        return lambda pred, target: cls(pred, target)[item]

    def __init__(self, pred, target):
        super().__init__()
        self["prediction"] = GridData(pred)
        self["target"] = GridData(target)

    def __missing__(self, key: str):
        value = self.calculate(key)
        self[key] = value
        return value

    images = {
        "square error density",
        "total weighted square error",
        "weighted square error vel_est",
        "weighted square error vel_unc",
        "weighted square error vel_mean",
        "weighted square error vel_std",
        "NLLL density",
        "total weighted NLLL",
        "weighted NLLL velocity",
        "weighted NLLL vel_est",
        # Per-axis mean NLLL (vel_mean channels 0,1 = grid vx, vy); vel_unc stays isotropic.
        "weighted NLLL vel_est_vx",
        "weighted NLLL vel_est_vy",
        "weighted NLLL vel_unc",
    }
    scalars = {*{"mean " + metric for metric in images}}

    def calculate(self, key: str):
        prefix, sep, remainder = key.partition(" ")

        if prefix in {"prediction", "target"}:
            if remainder == "vel_est":
                return self[prefix + sep + "vel_mean"]
            if remainder == "vel_unc":
                return self[prefix + sep + "vel_std"]
            if remainder == "opt_spd":
                return (self["square" + sep + prefix + sep + "vel_mean"]).sqrt()
            if remainder == "min_cost_per_dist":
                return self[prefix + sep + "density"] * 0.0
            return getattr(self[prefix], remainder)

        if prefix == "error":
            return self["prediction" + sep + remainder] - self["target" + sep + remainder]
        if prefix == "abs":
            return self[remainder].abs().sum(dim=-3, keepdim=True)
        if prefix == "square":
            return self[remainder].square().sum(dim=-3, keepdim=True)
        if prefix == "root":
            return self[remainder].sqrt()
        if prefix == "weighted":
            if "density" in remainder:
                return self[remainder]
            return self["target density"] * self[remainder]

        if prefix == "mean":
            return self[remainder].mean()
        if prefix == "sum":
            return self[remainder].sum()

        if prefix == "total":
            return (
                self[remainder + sep + "density"]
                + self[remainder + sep + "vel_est"]
                + self[remainder + sep + "vel_unc"]
            )

        if prefix == "KL":
            if remainder == "density":
                value = self["target density"] * -self["error logdensity"] + self["error density"]
                value[self["target density"] == 0] = 0
                return value
            if remainder == "vel_est":
                return 0.5 * self["square error vel_mean"]
            if remainder == "vel_unc":
                return 0.0 * self["target density"]
            if remainder == "velocity":
                return self[prefix + sep + "vel_est"]

        if prefix == "NLLL":
            if remainder == "density":
                return self["prediction density"] - self["target density"] * self["prediction logdensity"]
            if remainder == "vel_est":
                return 0.5 * self["square error vel_mean"]
            if remainder == "vel_est_vx":
                e = self["error vel_mean"][..., 0:1, :, :]
                return 0.5 * e.square()
            if remainder == "vel_est_vy":
                e = self["error vel_mean"][..., 1:2, :, :]
                return 0.5 * e.square()
            if remainder == "vel_unc":
                return 0.0 * self["target density"]
            if remainder == "velocity":
                return self[prefix + sep + "vel_est"] + self[prefix + sep + "vel_unc"]

        if prefix == "timewise":
            next_op, _, remainder = remainder.partition(" ")
            tensor: torch.Tensor = self[remainder]
            dims = set(range(tensor.ndim)) - {1}
            if next_op == "mean":
                return tensor.mean(dim=dims)
            if next_op == "min":
                return tensor.min(dim=dims).values
            if next_op == "max":
                return tensor.max(dim=dims).values

        raise KeyError(key)
