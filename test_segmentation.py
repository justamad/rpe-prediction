from src.devices import AzureKinect, plot_trajectories_for_all_joints
from src.processing import segment_exercises_based_on_joint, calculate_positions_std, calculate_velocity_std, calculate_acceleration_std

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")

means = []
exemplar = np.loadtxt("exe.np")
features = {'acc_std': ("Acceleration Std", calculate_acceleration_std, []),
            'velocity_std': ("Velocity Std", calculate_velocity_std, []),
            'pos_std': ("Positions std", calculate_positions_std, [])}

# Iterate over all trials
for counter in range(21):
    azure = AzureKinect(f"data/bjarne_trial/azure/{counter + 1:02}_sub")
    azure.process_raw_data()
    azure.filter_data()
    pelvis = azure.position_data['pelvis (y)'].to_numpy()
    pelvis = (pelvis - np.mean(pelvis)) / pelvis.std()

    repetitions, costs = segment_exercises_based_on_joint(-pelvis, exemplar, 30, 0.5, show=False)

    for t1, t2 in repetitions[:12]:
        df = azure.position_data.iloc[t1:t2, :]

        # Calculate features
        for feature, (_, method, means) in features.items():
            means.append(method(df))


# Plot the final figures
for feature, (title, _, means) in features.items():
    df = pd.concat(means)
    plot_trajectories_for_all_joints(df, title)
