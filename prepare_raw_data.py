import pandas as pd

from src.config import (
    SubjectDataIterator,
    ECGLoader,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
)

from src.camera import StereoAzure

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


iterator = SubjectDataIterator("data/raw").add_loader(StereoAzureSubjectLoader).add_loader(ECGLoader)
for trial in iterator.iterate_over_all_subjects():
    azure_paths = trial['azure']
    azure = StereoAzure(master_path=azure_paths['master'], sub_path=azure_paths['sub'])
    df = azure.fuse_cameras(show=False)
    df.index = pd.to_datetime(df.index, unit="s")

    imu = trial['ecg'][1]
    hr = trial['ecg'][2]

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle(trial['subject_name'])
    axs[0].plot(hr)
    axs[1].plot(imu)
    axs[2].plot(df['PELVIS (y)'])
    plt.show()
