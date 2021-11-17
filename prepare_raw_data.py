from src.camera import StereoAzure
from src.processing import resample_data, apply_butterworth_filter, calculate_magnitude

from src.config import (
    SubjectDataIterator,
    ECGLoader,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


iterator = SubjectDataIterator("data/raw").add_loader(StereoAzureSubjectLoader).add_loader(ECGLoader)
for trial in iterator.iterate_over_specific_subjects("9AE368"):
    azure_paths = trial['azure']
    azure = StereoAzure(master_path=azure_paths['master'], sub_path=azure_paths['sub'])
    df = azure.fuse_cameras(show=False)
    df.index = pd.to_datetime(df.index, unit="s")

    # IMU data
    imu = trial['ecg'][1]
    imu = imu.drop(columns=['Acceleration Magnitude'])
    imu = apply_butterworth_filter(imu, cutoff=4, order=4, sampling_rate=100)
    imu_mag = calculate_magnitude(imu)

    # Kinect data
    spine_chest = df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']]
    spine_chest = resample_data(spine_chest, 30, 100)
    spine_chest_acc = np.gradient(np.gradient(spine_chest.to_numpy(), axis=0), axis=0)
    spine_chest_mag = calculate_magnitude(spine_chest_acc)

    # hr = trial['ecg'][2]

    imu = imu.to_numpy()

    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.suptitle(trial['subject_name'])
    axs[0].plot(imu, label=['X', 'Y', 'Z'])
    axs[1].plot(imu_mag, label=['X', 'Y', 'Z'])
    axs[2].plot(spine_chest)
    axs[3].plot(spine_chest_acc)
    axs[4].plot(spine_chest_mag)
    plt.legend()
    plt.show()
