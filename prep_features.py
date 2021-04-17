from src.devices import AzureKinect
from src.features import calculate_features_sliding_window
from src.plot import plot_sensor_data_for_axes, plot_sensor_data_for_single_axis
# from src.config import ConfigReader
from src.processing import filter_dataframe

import matplotlib
matplotlib.use("TkAgg")

# means = []
excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]

# config = ConfigReader("data/bjarne_trial/config.json")
X = []
y = []

# Iterate over all trials
for set_data in range(9):
    # Create Azure Kinect objects and preprocess
    azure = AzureKinect(f"data/arne_flywheel/azure/{set_data + 1:02}_sub")
    azure.process_raw_data()
    azure.filter_data(order=4)
    positions = filter_dataframe(azure.position_data, excluded_joints)
    features = calculate_features_sliding_window(positions, window_size=60, step_size=1)
    plot_sensor_data_for_axes(features, "Feature")
