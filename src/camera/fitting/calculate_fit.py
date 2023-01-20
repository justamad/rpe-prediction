from .solver import fit_individual_frame
from typing import Dict, Tuple
from functools import partial
from tqdm import tqdm
from PyMoCapViewer import MoCapViewer

from .kinematic_model import (
    KINEMATIC_CHAIN,
    complete_restricted_angle_vector_with_zeros,
    calculate_bone_segments,
    restructure_data_frame,
    N_ANGLES,
)

import pandas as pd
import numpy as np
import logging
import multiprocessing as mp


def fit_inverse_kinematic_sequential(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bone_lengths = calculate_bone_segments(df)
    df = restructure_data_frame(df)

    fitted_skeleton = []
    fitted_angles = []
    opt_angles = np.zeros(N_ANGLES)

    logging.info(f"Fitting {df.shape[0]} frames using parallel mode.")

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
    pool = mp.Pool(processes=num_processes)

    results = pool.imap(partial(fit_individual_frame, bone_lengths=bone_lengths), data)
    for pos, opt_angles in tqdm(results):
        fitted_skeleton.append(pos.reshape(-1))
        fitted_angles.append(complete_restricted_angle_vector_with_zeros(opt_angles))

    fitted_skeleton = pd.DataFrame(data=np.array(fitted_skeleton), columns=df.columns)
    fitted_angles = pd.DataFrame(data=np.array(fitted_angles), columns=df.columns)
    return fitted_skeleton, fitted_angles


if __name__ == '__main__':
    test_df = pd.read_csv("../../fused.csv", index_col=0)
    new = False

    if new:
        positions, orientations = fit_inverse_kinematic_parallel(test_df)
        positions.to_csv("positions.csv", sep=";")
        orientations.to_csv("orientation.csv", sep=";")
    else:
        positions = pd.read_csv("positions.csv", index_col=0, sep=";")
        orientations = pd.read_csv("orientation.csv", index_col=0, sep=";")
        print(positions.shape)
        print(orientations.shape)

    viewer = MoCapViewer(sphere_radius=0.015, grid_axis=None)
    viewer.add_skeleton(
        positions,
        skeleton_orientations=orientations,
        orientation="euler",
        skeleton_connection=KINEMATIC_CHAIN
    )
    # viewer.add_skeleton(test_df)
    viewer.show_window()
