import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as md

from src.plot import plot_settings as ps
from argparse import ArgumentParser
from PySkeletonFitter import fuse_multiple_skeletons, fit_inverse_kinematic_parallel
from os.path import join


def plot_sync_data(pos_df: pd.DataFrame, imu_df: pd.DataFrame):
    # spine_chest = apply_butterworth_1d_signal(pos_df["SPINE_CHEST (y)"], cutoff=12, sampling_rate=30, order=4)
    # pos_grad = np.gradient(np.gradient(spine_chest, axis=0), axis=0)
    # pos_grad = pd.Series(pos_grad, index=pos_df.index)
    pos_shift_df = pos_df.copy()
    pos_shift_df.index = pos_shift_df.index + pd.Timedelta(seconds=-3.5)

    xfmt = md.DateFormatter("%M:%S")
    fix, axs = plt.subplots(2, 1, figsize=(ps.TEXT_WIDTH_INCH, ps.TEXT_WIDTH_INCH * 0.6), dpi=ps.DPI)

    # Original
    lns1 = axs[0].plot(imu_df["CHEST_ACCELERATION_Z"], color="black", label="CHEST IMU")
    ax2 = axs[0].twinx()
    lns2 = ax2.plot(-pos_shift_df["SPINE_CHEST (y)"], color="gray", label="Azure Kinect")
    ax2.set_ylabel("Position [m]")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    # Synchronized
    lns1 = axs[1].plot(imu_df["CHEST_ACCELERATION_Z"], label=f"CHEST IMU", color="black")
    ax2 = axs[1].twinx()
    lns2 = ax2.plot(-pos_df["SPINE_CHEST (y)"], label=f"Azure Kinect", color="gray")
    ax2.set_ylabel("Position [m]")

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    axs[0].xaxis.set_major_formatter(xfmt)
    axs[1].xaxis.set_major_formatter(xfmt)

    axs[0].set_xlabel("Time (MM:SS)")
    axs[0].set_ylabel("Acceleration [m/$s^2$]")
    axs[1].set_xlabel("Time (MM:SS)")
    axs[1].set_ylabel("Acceleration [m/$s^2$]")

    plt.tight_layout()
    plt.savefig("sync.pdf", bbox_inches='tight', pad_inches=ps.CUT_OFF, dpi=ps.DPI)


def fusing_plots(master_path: str, sub_path: str, joint_col: str, src_path: str):
    master_df = pd.read_csv(master_path, sep=";", index_col=0)
    sub_df = pd.read_csv(sub_path, sep=";", index_col=0)
    df1, df2, fused_df = fuse_multiple_skeletons(master_df, sub_df)
    fused_df.index = df1.index
    colors = ps.get_colors(3)

    plt.figure(figsize=(ps.TEXT_WIDTH_INCH, ps.TEXT_WIDTH_INCH * 0.5), dpi=ps.DPI)
    plt.plot(df1[joint_col], label="Left", color=colors[0])
    plt.plot(df2[joint_col], label="Right", color=colors[1])
    plt.plot(fused_df[joint_col], label="Fused", color=colors[2])

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(join(src_path, "fusing.pdf"), bbox_inches='tight', pad_inches=ps.CUT_OFF, dpi=ps.DPI)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="../../plots")
    args = parser.parse_args()

    matplotlib.use("WebAgg")
    fusing_plots(
        "../../../../PERSIST/subject_04/azure/master_1/positions_3d.csv",
        "../../../../PERSIST/subject_04/azure/sub_1/positions_3d.csv",
        joint_col="KNEE_RIGHT (y)",
        src_path=args.src_path,
    )

    pos_df = pd.read_csv("../../data/processed/37A7AA/10_set/pos.csv", index_col=0)
    imu_df = pd.read_csv("../../data/processed/37A7AA/10_set/imu.csv", index_col=0)
    pos_df.index = pd.to_datetime(pos_df.index)
    imu_df.index = pd.to_datetime(imu_df.index)
    plot_sync_data(pos_df, imu_df)
