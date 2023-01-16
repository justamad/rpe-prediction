from .kinematic_model import build_skeleton
from scipy.optimize import least_squares
from typing import Dict, Tuple

import numpy as np
import math

skeleton_history = []
orientation_history = []

lower_bounds = np.array([-180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180, -180])
upper_bounds = np.array([180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180])


def fit_individual_frame(
        frame: np.ndarray,
        bone_lengths: Dict,
        x0_angles: np.ndarray = np.zeros(21)
) -> Tuple[np.ndarray, np.ndarray]:
    bounds = (lower_bounds * math.pi / 180.0, upper_bounds * math.pi / 180.0)

    optimization = least_squares(
        optimize_skeleton_func,
        x0=x0_angles,
        method="trf",
        bounds=bounds,
        kwargs={"bone_lengths": bone_lengths, "gt_skeleton": frame},
        # verbose=2,
    )

    fitted_angles = optimization.x
    skeleton = build_skeleton(fitted_angles, bone_lengths)
    skeleton = skeleton + frame[0]
    return skeleton, fitted_angles


def optimize_skeleton_func(angles: np.ndarray, bone_lengths: Dict, gt_skeleton: np.ndarray) -> np.ndarray:
    fit_skeleton = build_skeleton(angles, bone_lengths)
    gt_skeleton = gt_skeleton - gt_skeleton[0]  # Centralize ground truth skeleton

    # skeleton_history.append(fit_global.reshape(1, -1))
    # orientation_history.append(kinematic_model.create_euler_matrices(angles))

    # per_joint_error.append(difference.mean(axis=1))
    # logging.info(f"RMSE: {np.sqrt(difference.mean())}")
    residuals = ((fit_skeleton - gt_skeleton) ** 2).reshape(-1)  # Return residuals as flat vector
    return residuals
