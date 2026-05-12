"""Minimal utilities vendored for PedPred models and GridData (no project_analysis)."""
from __future__ import annotations

import inspect
from functools import wraps
from typing import Sequence


class decorator_dict(dict):
    """Dict with ``.add(key)`` decorator to register torch overrides."""

    def add(self, key):
        assert key not in self

        def decorator(func):
            self[key] = func
            return func

        return decorator


class Exporter:
    def __init__(self):
        global __all__
        __all__ = []

    def __call__(self, o):
        global __all__
        __all__.append(o.__name__)
        return o


def seq_to_seq(func):
    """Broadcast a function across sequence (tuple, list) inputs."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        callargs = inspect.getcallargs(func, *args, **kwargs)
        lengths = {k: len(v) if isinstance(v, Sequence) else 0 for k, v in callargs.items()}
        max_len = max(lengths.values())
        assert all(l == 0 or l == max_len for l in lengths.values())
        if max_len:
            return tuple(
                func(
                    **{
                        k: v[i] if lengths[k] else v
                        for k, v in callargs.items()
                    }
                )
                for i in range(max_len)
            )
        return func(**callargs)

    return wrapper
