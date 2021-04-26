from src.devices import AzureKinect
from src.features import calculate_features_sliding_window
from src.plot import plot_sensor_data_for_axes
from src.config import SubjectSplitIterator
from src.processing import filter_dataframe, normalize_mean, segment_exercises_based_on_joint

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]
file_iterator = SubjectSplitIterator("data", train_percentage=0.7)


def calculate_kinect_data(iterator, mode="train"):
    X, y = [], []

    # Iterate over all trials
    for set_data in iterator:
        # Create Azure Kinect objects and preprocess
        azure = AzureKinect(set_data['azure'])
        azure.process_raw_data()
        azure.filter_data(order=4)
        positions = filter_dataframe(azure.position_data, excluded_joints)
        # segment_exercises_based_on_joint(positions[], )
        # orientations = filter_dataframe(azure.orientation_data, excluded_joints)
        # features = calculate_features_sliding_window(positions, window_size=60, step_size=1)
        features = normalize_mean(positions)
        plot_sensor_data_for_axes(features, "all")
        X.append(features)
        y.extend([set_data['rpe'] for _ in range(len(features))])

    X = pd.concat(X)
    y = pd.DataFrame(np.array(y), columns=["rpe"])

    print(X.shape)
    print(y.shape)
    X.to_csv(f"X_{mode}.csv", sep=";", index=False)
    y.to_csv(f"y_{mode}.csv", sep=";", index=False)


calculate_kinect_data(file_iterator.get_subject_iterator("train"), "train")
calculate_kinect_data(file_iterator.get_subject_iterator("test"), "test")
