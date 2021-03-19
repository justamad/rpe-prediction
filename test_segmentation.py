from src.devices import AzureKinect
from src.processing import segment_exercises_based_on_joint

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

means = []
exemplar = np.loadtxt("exe.np")

for counter in range(21):
    azure = AzureKinect(f"data/bjarne_trial/azure/{counter + 1:02}_sub")
    azure.process_raw_data()
    azure.filter_data()
    pelvis = azure.position_data['pelvis (y)'].to_numpy()
    pelvis = (pelvis - np.mean(pelvis)) / pelvis.std()

    repetitions, costs = segment_exercises_based_on_joint(-pelvis, exemplar, 30, 0.5, show=False)
    print(len(repetitions))

    for t1, t2 in repetitions[:12]:
        pelvis = azure.position_data['pelvis (y)'].to_numpy()
        repetition = pelvis[t1:t2]
        # velocity = np.gradient(repetition)
        # acceleration = np.gradient(velocity)
        means.append(repetition.std())

N = 11
cumsum, moving_aves = [0], []

for i, x in enumerate(means, 1):
    cumsum.append(cumsum[i - 1] + x)
    if i >= N:
        moving_ave = (cumsum[i] - cumsum[i - N]) / N
        moving_aves.append(moving_ave)

plt.title("Pelvis Acceleration Std Dev")
plt.plot(means, label="Pelvis Acc std")
plt.plot(moving_aves, label=f"moving average N={N}")
plt.legend()
plt.show()
