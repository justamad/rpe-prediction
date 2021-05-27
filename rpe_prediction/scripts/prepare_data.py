from rpe_prediction.devices import AzureKinect
from rpe_prediction.features import calculate_features_sliding_window
from rpe_prediction.config import SubjectDataIterator, RawDataLoaderSet
from rpe_prediction.processing import filter_dataframe, normalize_mean

import numpy as np
import pandas as pd

excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]
file_iterator = SubjectDataIterator("../../data/raw", RawDataLoaderSet())


def prepare_skeleton_data(iterator, window_size=30, step_size=2):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param iterator: RawFileIterate that delivers all sets over all subjects
    @param window_size: The number of sampled in one window
    @param step_size: the step size the window is moved over time series
    @return: Tuple that contains input data and labels (input, labels)
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
        data = filter_dataframe(data, ["ori"])  # TODO: Add timestamps to position_data?

        # Normalize each set
        data = normalize_mean(data, std_dev_factor=1.0)

        features = calculate_features_sliding_window(data, window_size=window_size, step_size=step_size)
        x_data.append(features)
        y_labels.extend([set_data['rpe'] for _ in range(len(features))])
        groups.extend([set_data['group'] for _ in range(len(features))])

    X = pd.concat(x_data, ignore_index=True)
    y = pd.DataFrame(np.stack([y_labels, groups], axis=1), columns=["rpe", "group"])
    return X, y


if __name__ == '__main__':
    X, y = prepare_skeleton_data(file_iterator.iterate_over_all_subjects(), window_size=60, step_size=5)

    X.to_csv("../../x.csv", index=False, sep=';')
    y.to_csv("../../y.csv", index=False, sep=';')
    print(X.shape)
    print(y.shape)
