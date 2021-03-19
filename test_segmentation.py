from src.devices import AzureKinect, plot_trajectories_for_all_joints
from src.processing import segment_exercises_based_on_joint, calculate_velocity

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

means = []
exemplar = np.loadtxt("exe.np")

# Iterate over all trials
for counter in range(21):
    azure = AzureKinect(f"data/bjarne_trial/azure/{counter + 1:02}_sub")
    azure.process_raw_data()
    azure.filter_data()
    pelvis = azure.position_data['pelvis (y)'].to_numpy()
    pelvis = (pelvis - np.mean(pelvis)) / pelvis.std()

    # example = pelvis[195:248]
    # np.savetxt("exe.np", example)
    # plt.plot(pelvis)
    # plt.show()

    repetitions, costs = segment_exercises_based_on_joint(-pelvis, exemplar, 30, 0.5, show=False)

    for t1, t2 in repetitions[:12]:
        df = azure.position_data.iloc[t1:t2, :]
        means.append(calculate_velocity(df))


df = pd.concat(means)
plot_trajectories_for_all_joints(df)
