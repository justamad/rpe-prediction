from rpe_prediction.config import SubjectDataIterator, RPELoader, FusedAzureLoader
from rpe_prediction.processing import compute_statistics_for_subjects
from .skeleton_features import calculate_3d_joint_velocities, calculate_joint_angles_with_reference_joint, \
    calculate_angles_between_3_joints
from .sliding_window import calculate_features_sliding_window

import pandas as pd
import numpy as np


def prepare_skeleton_data(input_path, window_size=30, overlap=0.5):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param input_path: the current path where data resides in
    @param window_size: The number of sampled in one window
    @param overlap: The current overlap in percent
    @return: Tuple that contains input data and labels (input, labels)
    """
    file_iterator = SubjectDataIterator(input_path).add_loader(RPELoader).add_loader(FusedAzureLoader)
    means, std_dev = compute_statistics_for_subjects(file_iterator.iterate_over_all_subjects())
    x_data = []
    y_data = []

    for set_data in file_iterator.iterate_over_all_subjects():
        subject_name = set_data['subject_name']
        kinect_data = pd.read_csv(set_data['azure'], sep=';', index_col=False).set_index('timestamp', drop=True)
        kinect_data = (kinect_data - means[subject_name]) / std_dev[subject_name]

        # Calculate and concatenate features
        velocity = calculate_3d_joint_velocities(kinect_data)
        angle_three = calculate_angles_between_3_joints(kinect_data)
        angle_origin = calculate_joint_angles_with_reference_joint(kinect_data)
        angle = pd.concat([angle_three, angle_origin], axis=1)

        angles_velocity = angle.diff(axis=0).dropna(axis='index')
        angles_velocity.rename(lambda x: x + "_SPEED", axis='columns', inplace=True)
        features = pd.concat([velocity, angle.iloc[1:], angles_velocity], axis=1)

        features = calculate_features_sliding_window(features.reset_index(), window_size=window_size, overlap=overlap)
        x_data.append(features)

        # Construct y-data with pseudonyms, rpe values and groups
        y = np.repeat([[set_data['subject_name'], set_data['rpe'], set_data['group'], set_data['nr_set']]],
                      len(features), axis=0)
        y_data.append(pd.DataFrame(y, columns=['name', 'rpe', 'group', 'set']))

    return pd.concat(x_data, ignore_index=True), pd.concat(y_data, ignore_index=True)
