import numpy as np
from pyedflib import highlevel
from tqdm import tqdm

import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

path = "../../../../Volumes/INTENSO/RPE_Data"


def calculate_hrv_features(
        ecg_signal: np.ndarray,
        ecg_sampling_rate: int,
        hrv_sampling_rate: int,
        hrv_window_size: int,
):
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=ecg_sampling_rate, method="neurokit")
    peaks, info = nk.ecg_peaks(ecg_clean, method="neurokit", sampling_rate=ecg_sampling_rate, correct_artifacts=True)
    peaks = np.array(peaks["ECG_R_Peaks"])

    df = pd.DataFrame()

    step_size = ecg_sampling_rate // hrv_sampling_rate
    win_size = hrv_window_size * ecg_sampling_rate

    for index in tqdm(range(0, len(ecg), step_size)):
        try:
            sub_array = peaks[index:index + win_size]
            hrv_time = nk.hrv_time(sub_array, sampling_rate=ecg_sampling_rate, show=True)
            df = pd.concat([df, hrv_time], ignore_index=False)

        except Exception as e:
            print(f"Error for window: {index} with message {e}")

    return df


for subject in os.listdir(path):
    signals, signal_headers, header = highlevel.read_edf(f"{path}/{subject}/ecg-{subject}.edf")
    print(signal_headers, header)
    df_edf = pd.DataFrame(data=signals[0], columns=["ecg"])
    df_edf.index = pd.to_datetime(df_edf.index, unit="ms")

    ecg = df_edf['ecg'].to_numpy()
    hrv_features = calculate_hrv_features(ecg, ecg_sampling_rate=1000, hrv_sampling_rate=4, hrv_window_size=30)
    hrv_features.to_csv("hrv.csv", index=False, sep=";")

    for column in hrv_features.columns:
        plt.plot(hrv_features[column], label=column)

    plt.legend()
    plt.show()

    # fig, axs = plt.subplots(3, 1, sharex=True)
    # fig.suptitle(subject)
    # axs[0].plot(hr_x, hr)
    # axs[1].plot(ecg)
    # axs[1].scatter(peaks, ecg[peaks])
    # axs[2].plot(ecg_clean)
    # axs[2].scatter(peaks, ecg_clean[peaks])
    # plt.show()
