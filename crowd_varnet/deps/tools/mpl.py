"""Matplotlib helpers for ``GridData.plot`` (circles for velocity std)."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle


def circles(x, y, s, c="b", vmin=None, vmax=None, **kwargs):
    """
    Scatter-style circles in *data* coordinates; returns a PatchCollection
    for ``ax.add_collection(...)``.
    Adapted from partial_observation_experiments/tools/mpl.py (BSD-3-Clause).
    """
    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None

    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_) for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    return collection
