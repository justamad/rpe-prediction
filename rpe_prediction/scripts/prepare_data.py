from rpe_prediction.devices import AzureKinect
from rpe_prediction.features import calculate_features_sliding_window
from rpe_prediction.plot import plot_sensor_data_for_axes
from rpe_prediction.config import RawDataIterator
from rpe_prediction.processing import filter_dataframe, normalize_mean, segment_exercises_based_on_joint

import numpy as np
import pandas as pd
import matplotlib

excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]
file_iterator = RawDataIterator("data/raw")


def prepare_skeleton_data(iterator):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param iterator: RawFileIterate that delivers all sets over all subjects
    @return:
    """
    x_data, y_labels = [], []
    groups = []

    # Iterate over all trials
    for set_data in iterator:
        # Create Azure Kinect objects and preprocess
        azure = AzureKinect(set_data['azure'])
        azure.process_raw_data(log=False)
        azure.filter_data(order=4)
        data = filter_dataframe(azure._data, excluded_joints)
        data = filter_dataframe(data, ["ori"])  # TODO: Fix, add timestamps to position_data?
        # data = normalize_mean(data, std_dev_factor=1.0)
        # plot_sensor_data_for_axes(data, "Position_data", azure.get_skeleton_joints(), f"{set_data['azure']}_pos.png")
        # segment_exercises_based_on_joint(positions[], )
        features = calculate_features_sliding_window(data, window_size=30, step_size=10)
        x_data.append(features)
        y_labels.extend([set_data['rpe'] for _ in range(len(features))])
        groups.extend([set_data['group'] for _ in range(len(features))])

    X = pd.concat(x_data, ignore_index=True)
    y = pd.DataFrame(np.stack([y_labels, groups], axis=1), columns=["rpe", "groups"])
    X.to_csv("x.csv", index=False, sep=';')
    y.to_csv("y.csv", index=False, sep=';')

    print(X.shape)
    print(y.shape)


if __name__ == '__main__':
    prepare_skeleton_data(file_iterator.iterate_over_all_subjects())
