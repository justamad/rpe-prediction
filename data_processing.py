from typing import List, Tuple
from argparse import ArgumentParser
from os.path import join, exists, isfile
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from cycler import cycler
from src.dataset import SubjectDataIterator, impute_dataframe, mask_repetitions
from src.features import CustomFeatures, calculate_skeleton_images

from src.processing import (
    segment_kinect_signal,
    apply_butterworth_filter,
    resample_data,
)

import numpy as np
import json
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt


def synchronize_flywheel_data(fw_durations: np.ndarray, azure_durations: np.ndarray) -> Tuple[List[bool], List[bool]]:
    if len(fw_durations) == len(azure_durations):
        return [True for _ in range(len(fw_durations))], [True for _ in range(len(azure_durations))]

    flywheel_mean = fw_durations / fw_durations.max()
    pos_mean = azure_durations / azure_durations.max()

    max_length = min(len(flywheel_mean), len(pos_mean))
    if len(flywheel_mean) > len(pos_mean):
        shift = calculate_cross_correlation_arrays(flywheel_mean, pos_mean)
        flywheel_mask = [False for _ in range(len(flywheel_mean))]
        flywheel_mask[shift:shift + max_length] = [True for _ in range(max_length)]
        return flywheel_mask, [True for _ in range(len(pos_mean))]

    shift = calculate_cross_correlation_arrays(pos_mean, flywheel_mean)
    pos_mask = [False for _ in range(len(pos_mean))]
    pos_mask[shift:shift + max_length] = [True for _ in range(max_length)]
    return [True for _ in range(len(flywheel_mean))], pos_mask


def calculate_cross_correlation_arrays(reference: np.ndarray, target: np.ndarray) -> int:
    reference = (reference - np.mean(reference)) / np.std(reference)
    target = (target - np.mean(target)) / np.std(target)

    diffs = []
    for shift in range(0, len(reference) - len(target) + 1):
        diffs.append(np.sum(np.abs(reference[shift:shift + len(target)] - target)))

    shift = np.argmin(diffs)
    return shift


def process_all_raw_data(src_path: str, dst_path: str, plot_path: str):
    iterator = SubjectDataIterator(
        base_path=src_path,
        dst_path=dst_path,
        data_loader=[
            SubjectDataIterator.IMU,
            SubjectDataIterator.HRV,
        ]
    )

    for set_id, trial in enumerate(iterator.iterate_over_all_subjects()):
        hrv_df = trial[SubjectDataIterator.HRV]
        imu_df = trial[SubjectDataIterator.IMU]

        imu_df = apply_butterworth_filter(df=imu_df, cutoff=20, order=4, sampling_rate=128)
        for df, name in zip([imu_df, hrv_df], ["imu", "hrv"]):
            df.to_csv(join(trial["dst_path"], f"{name}.csv"))


def iterate_segmented_data(src_path: str, mode: str, plot: bool = False, plot_path: str = None):
    if not exists(src_path):
        raise FileNotFoundError(f"Could not find source path {src_path}")

    for subject in os.listdir(src_path):
        rpe_file = join(src_path, subject, "rpe_ratings.json")
        if not isfile(rpe_file):
            raise FileNotFoundError(f"Could not find RPE file for subject {subject}")

        with open(rpe_file) as f:
            rpe_values = json.load(f)
        rpe_values = {k: v for k, v in enumerate(rpe_values["rpe_ratings"])}

        subject_plot_path = join(plot_path, f"segmented_{mode}", subject)
        if not exists(subject_plot_path):
            os.makedirs(subject_plot_path)

        subject_path = join(src_path, subject)
        set_folders = list(filter(lambda x: x != "rpe_ratings.json", os.listdir(subject_path)))
        set_folders = sorted(map(lambda x: (int(x.split("_")[0]), join(subject_path, x)), set_folders))
        for set_id, set_folder in set_folders:
            logging.info(f"Processing subject {subject}, set {set_id}")

            if set_id not in rpe_values:
                logging.error(f"Something is odd with {subject} and {set_id}")
                continue

            def read_and_process_dataframe(target: str):
                df = pd.read_csv(join(set_folder, f"{target}.csv"), index_col=0)
                df.index = pd.to_datetime(df.index)
                return df

            dataframes = [read_and_process_dataframe(target) for target in ["imu", "hrv"]]
            imu_df, hrv_df = dataframes

            part_repetitions, full_repetitions = segment_kinect_signal(
                -imu_df["CHEST_ACCELERATION_Z"],
                prominence=0.01,
                std_dev_p=0.4,
                min_dist_p=0.5,
                min_time=30,
                mode=mode,
                show=False,
            )
            if len(full_repetitions) == 0:
                logging.warning(f"No repetitions found for subject {subject}, set {set_id}")
                continue

            imu_df = apply_butterworth_filter(df=imu_df, cutoff=16, order=4, sampling_rate=128)

            # Mask all repetitions to delete the ones that are not full
            imu_df = mask_repetitions(imu_df, full_repetitions, col_name="Repetition")
            hrv_df = mask_repetitions(hrv_df, full_repetitions, col_name="Repetition")

            imu_df = mask_repetitions(imu_df, part_repetitions, col_name="Repetition")

            if plot:
                fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 12), sharex="col")

                axs[0].set_title("IMU Chest Acceleration (z)")
                axs[0].plot(imu_df["CHEST_ACCELERATION_Z"], color="gray")
                for p1, p2 in part_repetitions:
                    axs[0].plot(imu_df["CHEST_ACCELERATION_Z"][p1:p2])

                axs[1].set_title("TRIMP Feature")
                axs[1].plot(hrv_df["Load (TRIMP)"], color="gray")
                for p1, p2 in full_repetitions:
                    axs[1].plot(hrv_df["Load (TRIMP)"][p1:p2])

                plt.savefig(join(subject_plot_path, f"{set_id}.png"))
                plt.clf()
                plt.cla()
                plt.close()

            # Truncate dataframes to valid repetitions
            imu_df = imu_df[imu_df["Repetition"] != -1]
            hrv_df = hrv_df[hrv_df["Repetition"] != -1]

            yield {
                "meta": {"rpe": rpe_values[set_id], "subject": subject, "set_id": set_id,},
                "imu_df": imu_df,
                "hrv_df": hrv_df,
            }


def prepare_segmented_data_for_ml(src_path: str, dst_path: str, mode: str, plot: bool = False, plot_path: str = None):
    final_df = pd.DataFrame()
    settings = CustomFeatures()

    for trial in iterate_segmented_data(src_path, mode=mode, plot=plot, plot_path=plot_path):
        meta_data, imu_df, hrv_df = trial.values()
        imu_features_df = extract_features(imu_df, column_id="Repetition", default_fc_parameters=settings)
        imu_features_df = impute(imu_features_df)  # Replace Nan and inf by with extreme values (min, max)
        hrv_mean = hrv_df.groupby("Repetition").mean()

        total_df = pd.concat(
            [
                imu_features_df.reset_index(drop=True).add_prefix(f"{mode.upper()}_PHYSILOG_"),
                hrv_mean.reset_index(drop=True).add_prefix("HRV_"),
            ], axis=1,
        )

        for key, value in meta_data.items():
            total_df[key] = value
        final_df = pd.concat([final_df, total_df], axis=0)

    final_df = impute_dataframe(final_df)
    final_df.reset_index(drop=True, inplace=True)
    final_df.to_csv(join(dst_path, f"{mode}_stat.csv"))


def prepare_segmented_data_for_dl(src_path: str, dst_path: str, plot: bool, plot_path: str):
    repetition_data = []
    skeleton_images = []
    for trial in iterate_segmented_data(src_path, mode="full", plot=plot, plot_path=plot_path):
        meta_data, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()
        flywheel_df = flywheel_df.add_prefix("FLYWHEEL_")

        s = "Repetition"
        reps = pos_df[s]
        skeleton_img = calculate_skeleton_images(pos_df, ori_df)
        for rep_count, rep_idx in enumerate(reps.unique()):
            skeleton_images.append(skeleton_img[reps == rep_idx])
            new_data = meta_data.copy()
            new_data.update(flywheel_df.iloc[rep_count].to_dict())
            new_data.update(hrv_df[hrv_df[s] == rep_idx].drop(columns=[s]).add_prefix("HRV_").mean().to_dict())
            repetition_data.append(new_data)

    skeleton_images = np.array(skeleton_images, dtype=object)
    np.savez(join(dst_path, f"X_seg.npz"), X=skeleton_images)

    final_df = pd.DataFrame(repetition_data)
    final_df.to_csv(join(dst_path, f"y_seg.csv"), index=False)


def prepare_data_dl_entire_trials(src_path: str, dst_path: str, plot: bool, plot_path: str, fuse: bool = False):
    skeleton_images = []
    imu_data = []
    labels = []
    for trial in iterate_segmented_data(src_path, mode="full", plot=plot, plot_path=plot_path):
        meta_dict, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()

        if fuse:
            pos_df = resample_data(pos_df, 30, 128)
            min_length = min(len(pos_df), len(imu_df))
            pos_df = pos_df.iloc[:min_length]
            imu_df = imu_df.iloc[:min_length]

        skeleton_img = calculate_skeleton_images(pos_df, ori_df)
        skeleton_images.append(skeleton_img)
        imu_data.append(imu_df.drop("Repetition", axis=1, inplace=False).values)
        hrv = hrv_df.mean().to_dict()
        labels.append({**meta_dict, **hrv})
        # break

    if fuse:
        fused_images = [np.concatenate([skeleton, imu], axis=1) for skeleton, imu in zip(skeleton_images, imu_data)]
        fused_images = np.array(fused_images, dtype=object)
        np.savez(join(dst_path, f"X_fused.npz"), X=fused_images)
    else:
        X = np.array(skeleton_images, dtype=object)
        np.savez(join(dst_path, f"X_kinect.npz"), X=X)
        X = np.array(imu_data, dtype=object)
        np.savez(join(dst_path, "X_imu.npz"), X=X)

    y = pd.DataFrame(labels)
    y.to_csv(join(dst_path, "y.csv"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, dest="raw_path", default="data/PERSIST")
    parser.add_argument("--proc_path", type=str, dest="proc_path", default="data/processed")
    parser.add_argument("--train_path", type=str, dest="train_path", default="data/training")
    parser.add_argument("--plot_path", type=str, dest="plot_path", default="plots")
    parser.add_argument("--show", type=bool, dest="show", default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # matplotlib.use("WebAgg")
    default_cycler = (cycler(color=['#FF007F', '#D62598']))
    plt.rc('axes', prop_cycle=default_cycler)

    os.makedirs(args.proc_path, exist_ok=True)
    os.makedirs(args.train_path, exist_ok=True)

    process_all_raw_data(args.raw_path, args.proc_path, args.plot_path)
    # prepare_segmented_data_for_ml(args.proc_path, args.train_path, mode="concentric", plot=args.show, plot_path=args.plot_path)
    # prepare_segmented_data_for_ml(args.proc_path, args.train_path, mode="eccentric", plot=args.show, plot_path=args.plot_path)
    prepare_segmented_data_for_ml(args.proc_path, args.train_path, mode="full", plot=args.show, plot_path=args.plot_path)

    # prepare_segmented_data_for_dl(args.proc_path, dst_path=args.train_path, plot=args.show, plot_path=args.plot_path)
    # prepare_data_dl_entire_trials(args.proc_path, dst_path=args.train_path, plot=args.show, plot_path=args.plot_path, fuse=True)
