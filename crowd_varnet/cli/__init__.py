"""CrowdVarNet 命令行入口（内部实现）。"""

from ._common import build_model_from_ckpt, load_training_meta

__all__ = ["build_model_from_ckpt", "load_training_meta"]
