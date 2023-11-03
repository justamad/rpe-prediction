import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.plot import plot_settings as ps
from src.processing import segment_kinect_signal
from src.processing import apply_butterworth_1d_signal


def plot_segmentation(pos_df: pd.DataFrame):
    pelvis = pos_df["PELVIS (y)"]
    pelvis_smooth = apply_butterworth_1d_signal(pelvis, cutoff=12, sampling_rate=30, order=4)
    pelvis_smooth = pd.Series(pelvis_smooth, index=pelvis.index)

    concentric, _ = segment_kinect_signal(
        pelvis, prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30, mode="concentric", show=False)
    eccentric, full_repetitions = segment_kinect_signal(
        pelvis, prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30, mode="eccentric", show=False)

    fig, (axs1, axs2) = plt.subplots(2, 1, figsize=(ps.text_width * ps.cm, ps.text_width * 0.5 * ps.cm), dpi=ps.dpi)
    axs1.plot(pelvis, color="black")
    axs1.set_ylabel("Distance (m)")
    axs1.set_xlabel("Time (s)")
    ymin, ymax = axs1.get_ylim()

    # Horizontal lines
    for start, end in full_repetitions:
        axs1.vlines(start, ymin=ymin, ymax=ymax, color="gray", alpha=1)
    axs1.vlines(full_repetitions[-1][1], ymin=ymin, ymax=ymax, color="gray", alpha=1)

    for i, (start, end) in enumerate(concentric):
        axs2.axvspan(start, end, color="0.1", alpha=0.3, label="Concentric" if i == 0 else None)

    for i, (start, end) in enumerate(eccentric):
        axs2.axvspan(start, end, color="0.7", alpha=0.3, label="Eccentric" if i == 0 else None)

    axs2.plot(pelvis_smooth, color="0.0")
    axs2.set_ylabel("Distance (m)")
    axs2.set_xlabel("Time (s)")
    axs2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("segmentation.pdf", dpi=ps.dpi)
    plt.close()


if __name__ == "__main__":
    matplotlib.use("WebAgg")
    pos_df = pd.read_csv("../../data/processed/927394/08_set/pos.csv", index_col=0)
    pos_df.index = pd.to_datetime(pos_df.index)
    plot_segmentation(pos_df)
