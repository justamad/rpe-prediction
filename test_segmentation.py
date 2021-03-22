from src.devices import AzureKinect
from src.processing import segment_exercises_based_on_joint, calculate_positions_std, calculate_velocity_std, calculate_acceleration_std, calculate_min_max_distance, calculate_acceleration_magnitude_std, filter_dataframe
from src.plot import plot_sensor_data_for_axes, plot_sensor_data_for_single_axis

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

means = []
exemplar = np.loadtxt("exe.np")
# features = {'acc_std': ("Acceleration Std", calculate_acceleration_std, []),
#             'velocity_std': ("Velocity Std", calculate_velocity_std, []),
#             'pos_std': ("Positions std", calculate_positions_std, []),
#             'min_max': ("Min-Max Distance", calculate_min_max_distance, [])}

features = {'acc_mag_std': ("Acceleration Magnitude Std", calculate_acceleration_magnitude_std, [])}

excluded_joints = ["eye", "ear", "nose", "wrist", "hand", "thumb"]

durations = []

# Iterate over all trials
for counter in range(21):
    azure = AzureKinect(f"data/bjarne_trial/azure/{counter + 1:02}_sub")
    azure.process_raw_data()
    azure.filter_data(order=4)
    pelvis = azure.position_data['pelvis (y)'].to_numpy()
    pelvis = (pelvis - np.mean(pelvis)) / pelvis.std()

    repetitions, costs = segment_exercises_based_on_joint(-pelvis, exemplar, 30, 0.5, show=False)
    positions = filter_dataframe(azure.position_data, excluded_joints)

    for t1, t2 in repetitions[:12]:
        df = positions.iloc[t1:t2, :]

        # Calculate features
        for feature, (_, method, means) in features.items():
            means.append(method(df))

        # Calculation duration of each repetition
        durations.append(abs(t2 - t1))


# Plot the final figures
for feature, (title, _, means) in features.items():
    df = pd.concat(means)
    plot_sensor_data_for_single_axis(df, title)

plt.plot(np.array(durations) / 30)
plt.ylabel("Time [s]")
plt.xlabel("Repetition No")
plt.title("Repetition Duration")
plt.tight_layout()
plt.show()
