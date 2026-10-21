"""
Centralized Immutable Configuration and Physical Constants for 4D Cell Tracking.
Author: Ayush Shukla (ayushshuklaivyleague@gmail.com)
"""

import os
from types import MappingProxyType
import numpy as np

# Physical Anisotropic Scales (Light-Sheet Fluorescence Microscopy)
# Axial (Z) resolution vs Lateral (XY) resolution
VOXEL_SCALE_Z: float = 1.625    # micrometers per voxel
VOXEL_SCALE_Y: float = 0.40625  # micrometers per voxel
VOXEL_SCALE_X: float = 0.40625  # micrometers per voxel

PHYSICAL_VOXEL_SCALE: np.ndarray = np.array([VOXEL_SCALE_Z, VOXEL_SCALE_Y, VOXEL_SCALE_X], dtype=np.float64)
ANISOTROPY_FACTOR: float = VOXEL_SCALE_Z / VOXEL_SCALE_X  # exactly 4.0:1

# Ground-Truth Metric Bipartite Tolerance
METRIC_MATCHING_RADIUS_UM: float = 7.0  # micrometers in physical space

# Centralized Immutable Configuration Dictionary
_RAW_CFG = {
    # Experiment Metadata
    "EXPERIMENT_TAG": "rank1_neural_linker_calibrated_divisions",
    "PRESET": "neural_linker_mitotic_assignment_v1",
    "SCORE_AXIS": "Rank 1: Neural-Aware Global Linker + Integrated Division Recovery + Node Calibration",

    # Model & Ensemble Configuration
    "USE_DUAL_SEED": True,
    "SECONDARY_EDGE_WEIGHT": 0.20,
    "SECONDARY_DETECTION_WEIGHT": 0.15,
    "DET_TTA": True,
    "DET_THRESHOLD": 0.965,

    # Bidirectional Harmonic Association Fusion
    "BIDIRECTIONAL_EDGE_WEIGHT": 0.15,
    "BIDIRECTIONAL_FUSION_MODE": "harmonic_probability",
    "DUAL_SEED_MIN_CANDIDATE_RETENTION": 0.90,

    # Node-Count Calibration (Upgrade #3)
    "CALIBRATE_NODE_COUNT": True,
    "TARGET_NODE_RATIO": 1.00,
    "MAX_NODE_THRESHOLD_SHIFT": 0.035,

    # Neural-Aware Global Linker (Upgrade #1)
    "CANDIDATE_RADIUS_UM": 8.8,          # Captures >99.1% empirical cell displacements
    "LEARNED_EDGE_WEIGHT": 4.0,           # Neural probability log-odds weight
    "DISTANCE_WEIGHT": 0.35,              # Smooth physical distance penalty
    "VELOCITY_WEIGHT": 0.50,              # EMA velocity consistency weight
    "VELOCITY_EMA_ALPHA": 0.60,
    "MIN_LINK_PROB": 0.02,
    "MAX_FRAME_NODES": 12000,

    # Integrated Mitotic Division Recovery (Upgrade #2)
    "ENABLE_DIVISIONS": True,
    "DIV_MAX_PARENT_UM": 10.5,            # Mother-to-daughter distance cap
    "DIV_MAX_SISTER_UM": 16.0,            # Sister-to-sister separation cap
    "DIV_MAX_COS_ANGLE": 0.50,            # Spindle angle threshold (relaxed from 0.20)
    "DIV_JOINT_BONUS": 1.25,              # Reward for consistent cytokinesis pair
    "DIV_REQUIRE_DIVERGENCE": False,      # Do not choke blastomeres lacking t+2 grandchildren
    "DEEPCENTER_DIV_VETO": False,         # Do not veto valid mitotic splits
    "DIV_GLOBAL_FRAC_CAP": 0.008,
    "DIV_FRAME_FRAC_CAP": 0.015,

    # Metric Integrity (Safeguarding bipartite precision)
    "ENABLE_GAP_CLOSING": False,          # Eliminates ~182 shortcut edges per video
    "ENABLE_GAP2_RECOVERY": False,
    "FILTER_SHORT_TRACKS": False,         # Preserves nascent daughter blastomeres
    "MIN_TRACK_LEN": 1,
    "PRUNE_ISOLATED": True,               # Cleans unconnected single-frame false detections
    "LINEFIT_SMOOTH": False,
}

# Immutable mapping proxy
CFG = MappingProxyType(_RAW_CFG)

# Synchronize to environment variables for external subprocesses
for k, v in CFG.items():
    env_key = f"BIOHUB_{k}"
    if isinstance(v, bool):
        os.environ[env_key] = "1" if v else "0"
    else:
        os.environ[env_key] = str(v)
