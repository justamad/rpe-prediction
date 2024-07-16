import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from src.plot import plot_settings as ps
from src.processing import segment_kinect_signal
from src.processing import apply_butterworth_1d_signal
from argparse import ArgumentParser
from os.path import join


def plot_segmentation(pos_df: pd.DataFrame, file_name: str):
    pelvis = pos_df["PELVIS (y)"]
    pelvis_smooth = apply_butterworth_1d_signal(pelvis, cutoff=12, sampling_rate=30, order=4)
    pelvis_smooth = pd.Series(pelvis_smooth, index=pelvis.index)

    concentric, _ = segment_kinect_signal(
        pelvis, prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30, mode="concentric", show=False)
    eccentric, full_repetitions = segment_kinect_signal(
        pelvis, prominence=0.01, std_dev_p=0.4, min_dist_p=0.5, min_time=30, mode="eccentric", show=False)

    colors = ps.get_colors(3)
    plt.subplots(figsize=(ps.TEXT_WIDTH_INCH, ps.TEXT_WIDTH_INCH * 0.45), dpi=ps.DPI)
    plt.plot(pelvis, color=colors[0])

    # axs1.set_ylabel("Distance (m)")
    # axs1.set_xlabel("Time (s)")
    # ymin, ymax = axs1.get_ylim()

    # Horizontal lines
    # for start, end in full_repetitions:
    #     axs1.vlines(start, ymin=ymin, ymax=ymax, color="gray", alpha=1)
    # axs1.vlines(full_repetitions[-1][1], ymin=ymin, ymax=ymax, color="gray", alpha=1)

    for i, (start, end) in enumerate(concentric):
        plt.axvspan(start, end, color=colors[1], alpha=0.3, label="Concentric" if i == 0 else None)

    for i, (start, end) in enumerate(eccentric):
        plt.axvspan(start, end, color=colors[2], alpha=0.3, label="Eccentric" if i == 0 else None)

    # axs2.plot(pelvis_smooth, color="0.0")
    plt.ylabel("Distance [cm]")
    plt.xlabel("Time [hh:mm:ss]")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(file_name, bbox_inches='tight', pad_inches=ps.CUT_OFF, dpi=ps.DPI)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="../../plots")
    args = parser.parse_args()

    pos_df = pd.read_csv("../../data/processed/927394/08_set/pos.csv", index_col=0)
    pos_df.index = pd.to_datetime(pos_df.index)
    plot_segmentation(pos_df, join(args.src_path, "segmentation.pdf"))
