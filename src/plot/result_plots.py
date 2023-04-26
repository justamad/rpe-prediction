from .plot_settings import column_width, cm, dpi
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr
from os.path import join, exists
from os import makedirs

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


y_labels = {
    "rpe": "RPE",
    "hr": "Mean Heart Rate (1/min",
    "powercon": "Power",
    "powerecc": "Power",
}


def evaluate_sample_predictions(result_dict: Dict, gt_column: str, file_name: str):
    fig, axes = plt.subplots(len(result_dict), sharey=True, figsize=(20, 40))

    rmse_all = []
    r2_all = []
    mape_all = []
    for idx, (subject_name, df) in enumerate(result_dict.items()):
        ground_truth = df[gt_column].to_numpy()
        predictions = df["predictions"].to_numpy()
        rmse = mean_squared_error(predictions, ground_truth, squared=False)
        r2 = r2_score(ground_truth, predictions)
        rmse_all.append(rmse)
        r2_all.append(r2)
        mape = mean_absolute_percentage_error(predictions, ground_truth)
        mape_all.append(mape)

        axes[idx].plot(ground_truth, label="Ground Truth")
        axes[idx].plot(predictions, label="Prediction")
        axes[idx].set_title(f"Subject: {subject_name}, RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}")

    fig.suptitle(
        f"RMSE: {np.mean(rmse_all):.2f} +- {np.std(rmse_all):.2f}, R2: {np.mean(r2_all):.2f} +- {np.std(r2_all):.2f}, MAPE: {np.mean(mape_all):.2f} +- {np.std(mape_all):.2f}")
    plt.legend()
    plt.savefig(file_name)
    # plt.show()
    plt.clf()
    plt.close()


def evaluate_aggregated_predictions(df: pd.DataFrame, gt_column: str, file_name: str):
    subjects = df["subject"].unique()
    fig, axes = plt.subplots(len(subjects), sharey=True, figsize=(20, 40))

    for idx, (subject_name) in enumerate(subjects):
        sub_df = df[df["subject"] == subject_name]

        labels = []
        pred = []

        for set_ids in sub_df["set_id"].unique():
            sub_sub_df = sub_df[sub_df["set_id"] == set_ids]
            ground_truth = sub_sub_df["ground_truth"].to_numpy().mean()
            # predictions = np.average(sub_sub_df["prediction"], weights=np.arange(len(sub_sub_df)))
            predictions = np.average(sub_sub_df["prediction"])
            labels.append(ground_truth)
            pred.append(predictions)

        axes[idx].plot(labels, label="Ground Truth")
        axes[idx].plot(pred, label="Prediction")

    plt.legend()
    # plt.savefig(file_name)
    plt.show()
    plt.clf()
    plt.close()


def evaluate_sample_predictions_individual(
        value_df: pd.DataFrame,
        exp_name: str,
        dst_path: str,
        pred_col: str = "prediction",
        gt_col: str = "ground_truth"
):
    for subject_name in value_df["subject"].unique():
        subject_df = value_df[value_df["subject"] == subject_name]
        ground_truth = subject_df[gt_col].values
        predictions = subject_df[pred_col].values

        r2 = r2_score(ground_truth, predictions)
        mape = mean_absolute_percentage_error(predictions, ground_truth)
        # mae = mean_absolute_error(predictions, ground_truth)
        sp, _ = spearmanr(ground_truth, predictions)
        rmse = mean_squared_error(predictions, ground_truth, squared=False)

        plt.figure(figsize=(column_width * cm, column_width * cm), dpi=dpi)
        plt.plot(ground_truth, label="Ground Truth")
        plt.plot(predictions, label="Prediction")
        plt.title(f"$R^2$={r2:.2f}, MAPE={mape:.2f}, Spearman={sp:.2f}")
        plt.xlabel("Repetition")
        # plt.ylabel(y_labels[exp_name])
        plt.legend()
        plt.tight_layout()

        if not exists(dst_path):
            makedirs(dst_path)

        plt.savefig(join(dst_path, f"{subject_name}.pdf"))
        # plt.show()
        plt.clf()
        plt.close()


def evaluate_nr_features(df: pd.DataFrame, dst_path: str):
    plt.figure(figsize=(column_width * cm, column_width * cm * 0.65), dpi=dpi)

    nr_features = sorted(df["n_features"].unique())
    for model in df["model"].unique():
        sub_df = df[df["model"] == model]

        x_axis = []
        y_axis = []
        errors = []
        for nr_feature in nr_features:
            sub_sub_df = sub_df[sub_df["n_features"] == nr_feature].sort_values(by="mean_test_r2", ascending=False).iloc[0]
            x_axis.append(nr_feature)
            y_axis.append(sub_sub_df["mean_test_r2"])
            errors.append(sub_sub_df["std_test_r2"])

        # plt.errorbar(x_axis, y_axis, yerr=errors, label=model.upper())
        plt.plot(x_axis, y_axis, label=model.upper())

    # plt.ylim(0, 1)
    plt.xticks(nr_features)
    plt.legend()
    plt.xlabel("Number of Features")
    plt.ylabel("$R^2$")
    plt.tight_layout()

    # plt.show()
    plt.savefig(join(dst_path, "nr_features.pdf"))
    plt.clf()
    plt.close()


def plot_subject_performance(df: pd.DataFrame, dst_path: str):
    metrics = []
    subjects = df["subject"].unique()
    for subject in df["subject"].unique():
        sub_df = df[df["subject"] == subject]
        metric, p_value = spearmanr(sub_df["ground_truth"], sub_df["prediction"])
        metrics.append(metric)
        print(f"{subject}: {metric:.2f} ({p_value:.2f})")

    fig, axs = plt.subplots(1, 1, figsize=(column_width * cm, column_width * cm / 2), dpi=dpi)
    axs.bar([f"{i+1:2d}" for i in range(len(subjects))], metrics)
    # plt.title("Spearman's Rho")
    plt.xlabel("Subjects")
    plt.ylabel("Spearman's Rho")
    plt.tight_layout()

    if not exists(dst_path):
        makedirs(dst_path)

    plt.savefig(join(dst_path, "subject_performance.pdf"))
    # plt.show()
    plt.clf()
    plt.close()
