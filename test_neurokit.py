from pyedflib import highlevel
from biosppy.signals.tools import get_heart_rate

import pandas as pd
import neurokit2 as nk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os


for subject in os.listdir("data/raw"):

    signals, signal_headers, header = highlevel.read_edf(f"data/raw/{subject}/ecg.edf")
    print(signal_headers, header)
    df_edf = pd.DataFrame(data=signals[0], columns=["ecg"])
    df_edf.index = pd.to_datetime(df_edf.index, unit="ms")

    ecg = df_edf['ecg'].to_numpy()

    ecg_clean = nk.ecg_clean(ecg, sampling_rate=1000, method='neurokit')
    _, rpeaks = nk.ecg_peaks(ecg_clean, method='neurokit', sampling_rate=1000, correct_artifacts=True)
    peaks = rpeaks['ECG_R_Peaks']

    hr_x, hr = get_heart_rate(peaks, sampling_rate=1000, smooth=False)

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle(subject)
    axs[0].plot(hr_x, hr)
    axs[1].plot(ecg)
    axs[1].scatter(peaks, ecg[peaks])
    axs[2].plot(ecg_clean)
    axs[2].scatter(peaks, ecg_clean[peaks])
    plt.show()
