from src.devices import AzureKinect, sample_data_uniformly
from src.processing import segment_exercises_based_on_joint, calculate_positions_std, calculate_velocity_std, calculate_acceleration_std, calculate_min_max_distance, calculate_acceleration_magnitude_std, filter_dataframe
from src.plot import plot_sensor_data_for_axes, plot_sensor_data_for_single_axis
from src.config import ConfigReader

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

means = []
exemplar = np.loadtxt("exe.np")
features = {'acc_std': ("Acceleration Std", calculate_acceleration_std, [], plot_sensor_data_for_axes),
            'velocity_std': ("Velocity Std", calculate_velocity_std, [], plot_sensor_data_for_axes),
            'pos_std': ("Positions std", calculate_positions_std, [], plot_sensor_data_for_axes),
            'min_max': ("Min-Max Distance", calculate_min_max_distance, [], plot_sensor_data_for_axes),
            'acc_mag_std': ("Acceleration Magnitude Std", calculate_acceleration_magnitude_std, [], plot_sensor_data_for_single_axis)}

excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]

config = ConfigReader("data/bjarne_trial/config.json")
durations = []

# Iterate over all trials
for set_data in range(9):
    # Create Azure Kinect objects and preprocess
    azure = AzureKinect(f"data/arne_flywheel/azure/{set_data + 1:02}_sub",)
    azure.process_raw_data()
    azure.filter_data(order=4)
    azure.resample_data(sampling_frequency=128)
    positions = filter_dataframe(azure.position_data, excluded_joints)

    # positions, new_x = sample_data_uniformly(positions, timestamps=azure.timestamps, sampling_rate=128)

    pelvis = positions['pelvis (y)'].to_numpy()
    pelvis = (pelvis - np.mean(pelvis)) / pelvis.std()

    repetitions, costs = segment_exercises_based_on_joint(pelvis, exemplar, 30, 0.5, show=True)

    for t1, t2 in repetitions[:12]:
        df = positions.iloc[t1:t2, :]

        # Calculate features
        for feature, (_, method, means, _) in features.items():
            means.append(method(df))

        # Calculation duration of each repetition
        durations.append(abs(t2 - t1))


# Plot the final figures
for feature, (title, _, means, plotting) in features.items():
    df = pd.concat(means)
    plotting(df, title)

plt.plot(np.array(durations) / 128)
plt.ylabel("Time [s]")
plt.xlabel("Repetition No")
plt.title("Repetition Duration")
plt.tight_layout()
plt.show()
