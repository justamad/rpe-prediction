import numpy as np

from src.config import SubjectDataIterator, ECGSubjectLoader, RPESubjectLoader

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


iterator = SubjectDataIterator(
    base_path="data/raw",
    log_path="data/results",
    loaders=[ECGSubjectLoader, RPESubjectLoader]
)

data = {}

for set_data in iterator.iterate_over_all_subjects():
    rpe = set_data["rpe"]
    hr = set_data["ecg"]["hr"]

    subject = set_data["subject_name"]
    tup_add = (rpe, np.max(hr), np.mean(hr), len(hr))
    if subject not in data:
        data[subject] = [tup_add]
    else:
        data[subject].append(tup_add)


for keys, values in data.items():
    fig, axs = plt.subplots(1, 4, sharex=True)
    axs[0].plot([x[0] for x in values])
    axs[1].plot([x[1] for x in values])
    axs[2].plot([x[2] for x in values])
    axs[3].plot([x[3] for x in values])
    plt.show()
    plt.close()
