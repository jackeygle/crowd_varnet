"""Legacy shim: CrowdVarNet no longer adds ``project_analysis`` to ``sys.path``."""
from __future__ import annotations

import warnings
from pathlib import Path


def ensure_project_analysis_path() -> Path:
    warnings.warn(
        "ensure_project_analysis_path() is a no-op; use crowd_varnet.deps only.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Path(__file__).resolve().parents[1]
