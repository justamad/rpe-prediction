from src.devices import AzureKinect
from src.features import calculate_features_sliding_window
from src.plot import plot_sensor_data_for_axes, plot_sensor_data_for_single_axis
from src.config import ConfigReader
from src.processing import filter_dataframe

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]
config = ConfigReader("data/justin/config.json")
X = []
y = []

# Iterate over all trials
for set_data in config.iterate_over_sets():
    # Create Azure Kinect objects and preprocess
    azure = AzureKinect(set_data['azure'])
    azure.process_raw_data()
    azure.filter_data(order=4)
    positions = filter_dataframe(azure.position_data, excluded_joints)
    features = calculate_features_sliding_window(positions, window_size=60, step_size=1)
    print(features.shape)
    # plot_sensor_data_for_axes(features, "Feature")
    X.append(features)
    y.extend([set_data['rpe'] for _ in range(len(features))])


X = pd.concat(X)
plot_sensor_data_for_axes(X, "all")
y = pd.DataFrame(np.array(y), columns=["rpe"])
print(X.shape)
print(y.shape)
X.to_csv("X.csv", sep=";", index=False)
y.to_csv("y.csv", sep=";", index=False)
