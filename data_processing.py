import numpy as np
import json
import os
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt

from typing import List, Tuple
from argparse import ArgumentParser
from os.path import join, exists, isfile
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tqdm import tqdm
from cycler import cycler
from src.dataset import SubjectDataIterator, zero_pad_array, impute_dataframe, mask_repetitions
from src.features import CustomFeatures, calculate_linear_joint_positions, calculate_skeleton_images
from src.processing import (
    segment_kinect_signal,
    apply_butterworth_filter,
    calculate_acceleration,
    calculate_cross_correlation_with_datetime,
)


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
            SubjectDataIterator.FLYWHEEL,
            SubjectDataIterator.AZURE,
            SubjectDataIterator.IMU,
            SubjectDataIterator.HRV,
        ]
    )

    for set_id, trial in enumerate(iterator.iterate_over_all_subjects()):
        pos_df, ori_df = trial[SubjectDataIterator.AZURE]
        flywheel_df = trial[SubjectDataIterator.FLYWHEEL]
        hrv_df = trial[SubjectDataIterator.HRV]
        imu_df = trial[SubjectDataIterator.IMU]

        imu_df = apply_butterworth_filter(df=imu_df, cutoff=20, order=4, sampling_rate=128)
        azure_acc_df = calculate_acceleration(pos_df)
        shift_dt = calculate_cross_correlation_with_datetime(
            reference_df=imu_df,
            ref_sync_axis="CHEST_ACCELERATION_Z",
            target_df=azure_acc_df,
            target_sync_axis="SPINE_CHEST (y)",
            show=False,
        )
        azure_acc_df.index += shift_dt
        pos_df.index += shift_dt
        ori_df.index += shift_dt

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 12))
        fig.suptitle(f"Subject: {trial['subject']}, Set: {trial['nr_set']}")
        axs[0].plot(pos_df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']])
        axs[0].set_title("Kinect Position")
        axs[1].plot(azure_acc_df[['SPINE_CHEST (x)', 'SPINE_CHEST (y)', 'SPINE_CHEST (z)']])
        axs[1].set_title("Kinect Acceleration")
        axs[2].plot(imu_df[['CHEST_ACCELERATION_X', 'CHEST_ACCELERATION_Y', 'CHEST_ACCELERATION_Z']])
        axs[2].set_title("Gaitup Acceleration")
        axs[3].plot(hrv_df[["Intensity (TRIMP/min)"]])
        axs[3].set_title("HRV")

        plt.savefig(join(plot_path, f"{trial['subject']}_{trial['nr_set']}.png"))
        # plt.show(block=True)
        plt.close()
        plt.cla()
        plt.clf()

        for df, name in zip([pos_df, ori_df, imu_df, hrv_df, flywheel_df], ["pos", "ori", "imu", "hrv", "flywheel"]):
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

            def read_and_process_dataframe(target: str):
                df = pd.read_csv(join(set_folder, f"{target}.csv"), index_col=0)
                df.index = pd.to_datetime(df.index)
                return df

            dataframes = [read_and_process_dataframe(target) for target in ["imu", "pos", "ori", "hrv", "flywheel"]]
            imu_df, pos_df, ori_df, hrv_df, flywheel_df = dataframes

            part_repetitions, full_repetitions = segment_kinect_signal(
                pos_df["PELVIS (y)"],
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

            pos_df = apply_butterworth_filter(df=pos_df, cutoff=16, order=4, sampling_rate=30)
            ori_df = apply_butterworth_filter(df=ori_df, cutoff=16, order=4, sampling_rate=30)
            imu_df = apply_butterworth_filter(df=imu_df, cutoff=16, order=4, sampling_rate=128)

            # Mask all repetitions to delete the ones that are not full
            pos_df = mask_repetitions(pos_df, full_repetitions, col_name="Repetition")
            ori_df = mask_repetitions(ori_df, full_repetitions, col_name="Repetition")
            imu_df = mask_repetitions(imu_df, full_repetitions, col_name="Repetition")
            hrv_df = mask_repetitions(hrv_df, full_repetitions, col_name="Repetition")

            pos_reps = pos_df["Repetition"].unique()
            imu_reps = imu_df["Repetition"].unique()
            if len(pos_reps) != len(imu_reps):
                logging.warning(f"Different nr of reps: {subject}, set {set_id}: {len(pos_reps)} vs. {len(imu_reps)}")
                continue

            pos_df = mask_repetitions(pos_df, part_repetitions, col_name="Repetition")
            ori_df = mask_repetitions(ori_df, part_repetitions, col_name="Repetition")
            imu_df = mask_repetitions(imu_df, part_repetitions, col_name="Repetition")

            # Synchronize sensors to Flywheel data
            flywheel_durations = list(flywheel_df["duration"])
            azure_durations = [(p2 - p1).total_seconds() for p1, p2 in full_repetitions]

            flywheel_mask, pos_mask = synchronize_flywheel_data(
                fw_durations=np.array(flywheel_durations),
                azure_durations=np.array(azure_durations),
            )
            # Remove invalid repetitions from all sensors
            for rep_counter, valid_rep in enumerate(pos_mask):
                if not valid_rep:
                    pos_df = pos_df[pos_df["Repetition"] != rep_counter]
                    ori_df = ori_df[ori_df["Repetition"] != rep_counter]
                    imu_df = imu_df[imu_df["Repetition"] != rep_counter]
                    hrv_df = hrv_df[hrv_df["Repetition"] != rep_counter]

            if plot:
                fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), sharex="col")
                fig.suptitle(f"FlyWheel: {len(flywheel_df)} vs. Kinect: {len(part_repetitions)}")

                axs[0, 0].plot(pos_df["PELVIS (y)"], color="gray")
                for p1, p2 in part_repetitions:
                    axs[0, 0].plot(pos_df["PELVIS (y)"][p1:p2])

                axs[1, 0].plot(imu_df["CHEST_ACCELERATION_Z"], color="gray")
                for p1, p2 in part_repetitions:
                    axs[1, 0].plot(imu_df["CHEST_ACCELERATION_Z"][p1:p2])

                axs[2, 0].plot(hrv_df["Load (TRIMP)"], color="gray")
                for p1, p2 in full_repetitions:
                    axs[2, 0].plot(hrv_df["Load (TRIMP)"][p1:p2])

                x_axis = np.arange(max(len(flywheel_mask), len(pos_mask)))
                if len(pos_mask) < len(flywheel_mask):
                    false_idx = [i for i, x in enumerate(flywheel_mask) if not x]
                    for i in false_idx:
                        azure_durations.insert(i, 0)

                elif len(flywheel_mask) < len(pos_mask):
                    false_idx = [i for i, x in enumerate(pos_mask) if not x]
                    for i in false_idx:
                        flywheel_durations.insert(i, 0)

                axs[0, 1].bar(x_axis - 0.2, flywheel_durations, 0.4)
                axs[0, 1].bar(x_axis + 0.2, azure_durations, 0.4)

                # plt.show()
                plt.savefig(join(subject_plot_path, f"{subject}_{set_id}.png"))
                plt.clf()
                plt.cla()
                plt.close()

            # Truncate dataframes to valid repetitions
            pos_df = pos_df[pos_df["Repetition"] != -1]
            ori_df = ori_df[ori_df["Repetition"] != -1]
            imu_df = imu_df[imu_df["Repetition"] != -1]
            hrv_df = hrv_df[hrv_df["Repetition"] != -1]

            yield {
                "meta": {"rpe": rpe_values[set_id], "subject": subject, "set_id": set_id,},
                "imu_df": imu_df,
                "pos_df": pos_df,
                "ori_df": ori_df,
                "hrv_df": hrv_df,
                "flywheel_df": flywheel_df[flywheel_mask],
            }


def prepare_segmented_data_for_ml(src_path: str, dst_path: str, mode: str, plot: bool = False, plot_path: str = None):
    final_df = pd.DataFrame()
    settings = CustomFeatures()

    for trial in iterate_segmented_data(src_path, mode=mode, plot=plot, plot_path=plot_path):
        meta_data, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()
        c_f = len(flywheel_df)
        c_p = len(pos_df["Repetition"].unique())
        c_i = len(imu_df["Repetition"].unique())
        if c_f != c_p != c_i:
            logging.warning(f"Different nr of reps: {subject}, set {set_id}: {c_f} vs. {c_p} vs. {c_i}")
            continue

        pos_df = calculate_linear_joint_positions(pos_df)

        imu_features_df = extract_features(imu_df, column_id="Repetition", default_fc_parameters=settings)
        imu_features_df = impute(imu_features_df)  # Replace Nan and inf by with extreme values (min, max)
        pos_features_df = extract_features(pos_df, column_id="Repetition", default_fc_parameters=settings)
        pos_features_df = impute(pos_features_df)
        ori_features_df = extract_features(ori_df, column_id="Repetition", default_fc_parameters=settings)
        ori_features_df = impute(ori_features_df)
        hrv_mean = hrv_df.groupby("Repetition").mean()

        total_df = pd.concat(
            [
                pos_features_df.reset_index(drop=True).add_prefix(f"{mode.upper()}_KINECTPOS_"),
                ori_features_df.reset_index(drop=True).add_prefix(f"{mode.upper()}_KINECTORI_"),
                flywheel_df.reset_index(drop=True).add_prefix("FLYWHEEL_"),
                imu_features_df.reset_index(drop=True).add_prefix("PHYSILOG_"),
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
        skeleton_df = calculate_skeleton_images(pos_df, ori_df)
        for rep_count, rep_idx in enumerate(reps.unique()):
            skeleton_images.append(skeleton_df[reps == rep_idx])
            meta_data.update(flywheel_df.iloc[rep_count].to_dict())
            meta_data.update(hrv_df[hrv_df[s] == rep_idx].drop(columns=[s]).add_prefix("HRV_").mean().to_dict())
            repetition_data.append(meta_data)

    lengths = [len(img) for img in skeleton_images]
    max_length = max(lengths)
    max_subject = repetition_data[np.argmax(lengths)]
    logging.info(f"Max Rep Length: {max_length} by subject {max_subject['subject']}, set {max_subject['set_id']}")

    # Normalize skeleton images
    min_i = np.array([img.reshape((img.shape[0]*img.shape[1], 3)).min(axis=0) for img in skeleton_images]).min(axis=0)
    max_i = np.array([img.reshape((img.shape[0]*img.shape[1], 3)).max(axis=0) for img in skeleton_images]).max(axis=0)

    for i in tqdm(range(len(skeleton_images))):
        skeleton_images[i] = zero_pad_array((skeleton_images[i] - min_i) / (max_i - min_i), max_length)

    pad_images = np.array(skeleton_images)
    np.save(join(dst_path, f"{max_length}.npy"), pad_images)

    final_df = pd.DataFrame(repetition_data)
    final_df.to_csv(join(dst_path, f"{max_length}.csv"), index=False)


def prepare_data_for_dl_sliding_window(src_path: str, dst_path: str, plot: bool, plot_path: str):
    skeleton_images = []
    labels = []
    i = 0
    for trial in iterate_segmented_data(src_path, mode="full", plot=plot, plot_path=plot_path):
        meta_dict, imu_df, pos_df, ori_df, hrv_df, flywheel_df = trial.values()

        pos_df.drop("Repetition", axis=1, inplace=True)
        ori_df.drop("Repetition", axis=1, inplace=True)
        skeleton_img = calculate_skeleton_images(pos_df, ori_df)
        skeleton_images.append(skeleton_img)
        labels.append(meta_dict)
        i += 1
        if i > 15:
            break

    y = pd.DataFrame(labels)
    y.to_csv(join(dst_path, "y_lstm.csv"))

    X = np.array(skeleton_images, dtype=object)
    for subject in y["subject"].unique():
        mask = y["subject"] == subject

        images = X[mask]
        min_i = np.array([img.reshape((img.shape[0]*img.shape[1], 3)).min(axis=0) for img in images]).min(axis=0)
        max_i = np.array([img.reshape((img.shape[0]*img.shape[1], 3)).max(axis=0) for img in images]).max(axis=0)

        for i in range(len(images)):
            images[i] = (images[i] - min_i) / (max_i - min_i)

        X[mask] = images

    np.savez(join(dst_path, "X_lstm.npz"), X=X)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_path", type=str, dest="raw_path", default="/media/ch/Data/RPE_Analysis")
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

    matplotlib.use("WebAgg")
    default_cycler = (cycler(color=['#FF007F', '#D62598']))
    plt.rc('axes', prop_cycle=default_cycler)

    if not exists(args.proc_path):
        os.makedirs(args.proc_path)

    if not exists(args.train_path):
        os.makedirs(args.train_path)

    # process_all_raw_data(args.raw_path, args.proc_path, args.plot_path)

    # prepare_segmented_data_for_ml(args.proc_path, args.train_path, mode="concentric", plot=args.show, plot_path=args.plot_path)
    # prepare_segmented_data_for_ml(args.proc_path, args.train_path, mode="eccentric", plot=args.show, plot_path=args.plot_path)
    # prepare_segmented_data_for_ml(args.proc_path, args.train_path, mode="full", plot=args.show, plot_path=args.plot_path)

    # prepare_segmented_data_for_dl(args.proc_path, dst_path=args.train_path, plot=args.show, plot_path=args.plot_path)
    prepare_data_for_dl_sliding_window(args.proc_path, dst_path=args.train_path, plot=args.show, plot_path=args.plot_path)
