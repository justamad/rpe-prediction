from src.devices import AzureKinect
from src.features import calculate_features_sliding_window
from src.plot import plot_sensor_data_for_axes
from src.config import RawDataIterator
from src.processing import filter_dataframe, normalize_mean, segment_exercises_based_on_joint

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]
file_iterator = RawDataIterator("data/raw")


def prepare_skeleton_data(iterator):
    """
    Prepare Kinect skeleton data using the RawFileIterator
    @param iterator: RawFileIterate that delivers all sets over all subjects
    @return:
    """
    x_data, y_labels = [], []

    # Iterate over all trials
    for set_data in iterator:
        # Create Azure Kinect objects and preprocess
        azure = AzureKinect(set_data['azure'])
        azure.process_raw_data()
        azure.filter_data(order=4)
        data = filter_dataframe(azure._data, excluded_joints)
        # data = filter_dataframe(data, ["timestamp", "pos"])
        data = normalize_mean(data, std_dev_factor=1.0)
        # plot_sensor_data_for_axes(data, "Position_data", azure.get_skeleton_joints(), f"{set_data['azure']}_pos.png")
        # segment_exercises_based_on_joint(positions[], )
        # orientations = filter_dataframe(azure.orientation_data, excluded_joints)
        features = calculate_features_sliding_window(data, window_size=30, step_size=2)
        x_data.append(features)
        y_labels.extend([set_data['rpe'] for _ in range(len(features))])

    X = pd.concat(x_data, ignore_index=True)
    y = pd.DataFrame(np.array(y_labels), columns=["rpe"])
    X.to_csv("x.csv", index=False, sep=';')
    y.to_csv("y.csv", index=False, sep=';')

    print(X.shape)
    print(y.shape)


prepare_skeleton_data(file_iterator.iterate_over_all_subjects())
