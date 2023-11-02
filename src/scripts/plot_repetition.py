import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.plot import plot_settings as ps
from src.processing import segment_kinect_signal
from src.processing import apply_butterworth_1d_signal


matplotlib.use("WebAgg")

pos_df = pd.read_csv("../../data/processed/927394/08_set/pos.csv", index_col=0)
pos_df.index = pd.to_datetime(pos_df.index)
pelvis = pos_df["PELVIS (y)"]
pelvis_smooth = apply_butterworth_1d_signal(pelvis, cutoff=12, sampling_rate=30, order=4)
pelvis_smooth = pd.Series(pelvis_smooth, index=pelvis.index)

concentric, _ = segment_kinect_signal(pelvis, prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30,
                                      mode="concentric", show=False)
eccentric, full_repetitions = segment_kinect_signal(pelvis, prominence=0.01, std_dev_p=0.4, min_dist_p=0.5,
                                                    min_time=30, mode="eccentric", show=False)

fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(ps.text_width, ps.text_width / 2), dpi=ps.dpi)
axs1.plot(pelvis, color="gray")
axs1.set_ylabel("Distance (m)")
axs1.set_xlabel("Time (s)")
ymin, ymax = axs1.get_ylim()

# horizontal lines
for start, end in full_repetitions:
    axs1.vlines(start, ymin=ymin, ymax=ymax, color="red", alpha=1)
axs1.vlines(full_repetitions[-1][1], ymin=ymin, ymax=ymax, color="red", alpha=1)

for i, (start, end) in enumerate(concentric):
    axs2.axvspan(start, end, color="red", alpha=0.3, label="Concentric Phases" if i == 0 else None)

for i, (start, end) in enumerate(eccentric):
    axs2.axvspan(start, end, color="blue", alpha=0.3, label="Eccentric Phases" if i == 0 else None)

axs2.plot(pelvis_smooth, color="gray")
axs2.set_ylabel("Distance (m)")
axs2.set_xlabel("Time (s)")
axs2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("segmentation.pdf")
