"""Benchmark configuration — shared constants for all experiments."""
from pathlib import Path
import numpy as np

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path("/scratch/work/zhangx29/crowd_varnet")
PO_ROOT = Path("/scratch/work/zhangx29/Partial_observation")
DATA_ROOT = Path("/scratch/work/zhangx29/data")

# Teacher checkpoints
OUR_TEACHER_CKPT = PROJECT_ROOT / "runs/pedpred_v13_gru_mid_nlll_17843926/checkpoints/free-pig_best.hkl"
OUR_TEACHER_ARCH = "pedpred3_gru_mid"

THEIR_TEACHER_CKPT = PO_ROOT / "apt-ibex_train_model_28D.pth"
THEIR_TEACHER_ARCH = "pedpred3"  # PedPred3 from Partial_observation

# CrowdVarNet best checkpoint
CVN_BEST_CKPT = PROJECT_ROOT / "runs/cvn_p3_v13_horizon_17847842/best.pt"

# ============================================================
# Grid / State
# ============================================================
GRID_SIZE = (36, 12)
STATE_SHAPE = (4, 36, 12)
STATE_DIM = int(np.prod(STATE_SHAPE))
NUM_CHANNELS = 4
CH_NAMES = ("rho", "vx", "vy", "var")

# ============================================================
# Observation model (shared between EnKF and CrowdVarNet)
# ============================================================
NUM_AGENTS = 3
SENSING_RANGE = 5
OBS_NOISE_STD = np.array([0.05690936, 0.31472941, 0.08616199, 0.00644609])

# ============================================================
# EnKF parameters (from their best config)
# ============================================================
ENKF_ENSEMBLE_SIZE = 400
ENKF_PROC_STD = np.array([0.02829307, 0.31263075, 0.12325809, 0.41680932])
ENKF_OBS_STD = OBS_NOISE_STD
ENKF_LOC_RADIUS = 5
ENKF_INFLATION = 1.0

# ============================================================
# Experiment settings
# ============================================================
NUM_STEPS = 300  # rollout length
WARMUP_STEPS = 5  # CrowdVarNet uses 5 GT frames as history warmup
RHO_MASK_THR = 0.05  # density support threshold for masked metrics
RANDOM_SEED = 42

# ============================================================
# Physical bounds (shared)
# ============================================================
CLIP_BOUNDS = {
    "density": (0.0, 5.0),
    "vx": (-5.0, 5.0),
    "vy": (-5.0, 5.0),
    "var": (0.0, 2.0),
}
