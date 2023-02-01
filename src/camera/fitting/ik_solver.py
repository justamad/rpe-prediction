from scipy.optimize import least_squares
from typing import Dict, Tuple
from functools import partial
from tqdm import tqdm

from .kinematic_model import (
    calculate_forwards_kinematics,
    complete_restricted_angle_vector_with_zeros,
    calculate_bone_segments,
    restructure_data_frame,
    boundaries,
)

import multiprocessing as mp
import pandas as pd
import numpy as np
import math


def fit_inverse_kinematic_sequential(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bone_lengths = calculate_bone_segments(df)
    df = restructure_data_frame(df)

    fitted_skeleton = []
    fitted_angles = []
    opt_angles = np.zeros(len(boundaries))

    for frame in tqdm(range(df.shape[0])):
        pos, opt_angles = fit_individual_frame(
            frame=df.iloc[frame].values.T.reshape(-1, 3),
            bone_lengths=bone_lengths,
            x0_angles=opt_angles,
        )
        fitted_skeleton.append(pos.reshape(-1))
        fitted_angles.append(complete_restricted_angle_vector_with_zeros(opt_angles))

    fitted_skeleton = pd.DataFrame(data=np.array(fitted_skeleton), columns=df.columns)
    fitted_angles = pd.DataFrame(data=np.array(fitted_angles), columns=df.columns)
    return fitted_skeleton, fitted_angles


def fit_inverse_kinematic_parallel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bone_lengths = calculate_bone_segments(df)
    df = restructure_data_frame(df)

    fitted_skeleton = []
    fitted_angles = []

    data = [df.iloc[frame].values.T.reshape(-1, 3) for frame in range(len(df))]
    num_processes = mp.cpu_count()
    print(num_processes)

    pool = mp.Pool(processes=num_processes)
    results = pool.imap(partial(fit_individual_frame, bone_lengths=bone_lengths), data)
    for pos, opt_angles in tqdm(results):
        fitted_skeleton.append(pos.reshape(-1))
        fitted_angles.append(complete_restricted_angle_vector_with_zeros(opt_angles))

    fitted_skeleton = pd.DataFrame(data=np.array(fitted_skeleton), columns=df.columns)
    fitted_angles = pd.DataFrame(data=np.array(fitted_angles), columns=df.columns)
    return fitted_skeleton, fitted_angles


def fit_individual_frame(
        frame: np.ndarray,
        bone_lengths: Dict,
        x0_angles=np.zeros(len(boundaries)),
) -> Tuple[np.ndarray, np.ndarray]:
    bounds = boundaries * math.pi / 180.0
    bounds = (bounds[:, 0], bounds[:, 1])

    optimization = least_squares(
        optimize_skeleton_func,
        x0=x0_angles,
        method="trf",
        bounds=bounds,
        kwargs={"bone_lengths": bone_lengths, "gt_skeleton": frame},
    )

    fitted_angles = optimization.x
    skeleton = calculate_forwards_kinematics(fitted_angles, bone_lengths)
    skeleton = skeleton + frame[0]
    return skeleton, fitted_angles


def optimize_skeleton_func(angles: np.ndarray, bone_lengths: Dict, gt_skeleton: np.ndarray) -> np.ndarray:
    fit_skeleton = calculate_forwards_kinematics(angles, bone_lengths)
    gt_skeleton = gt_skeleton - gt_skeleton[0]  # Centralize ground truth skeleton
    return (fit_skeleton - gt_skeleton).reshape(-1)
