from .plot_settings import column_width, cm, dpi
from typing import Dict
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
import pandas as pd
import numpy as np

y_labels = {
    "rpe": "RPE",
    "hr": "Mean Heart Rate (1/min",
    "powerCon": "Power",
    "powerEcc": "Power",
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


def evaluate_aggregated_predictions(result_dict: Dict, gt_column: str, file_name: str):
    fig, axes = plt.subplots(len(result_dict), sharey=True, figsize=(20, 40))

    rmse_all = []
    r2_all = []
    pcc_all = []
    for idx, (subject_name, df) in enumerate(result_dict.items()):
        mean_df = df.groupby("set_id").mean(numeric_only=True)
        std_df = df.groupby("set_id").std(numeric_only=True)

        ground_truth = mean_df[gt_column].to_numpy()
        predictions = mean_df["predictions"].to_numpy()
        errors = std_df["predictions"].to_numpy()
        rmse = mean_squared_error(predictions, ground_truth, squared=False)
        r2 = r2_score(ground_truth, predictions)
        pcc = pearsonr(ground_truth, predictions)[0]
        rmse_all.append(rmse)
        r2_all.append(r2)
        pcc_all.append(pcc)

        axes[idx].plot(ground_truth, label="Ground Truth")
        axes[idx].errorbar(np.arange(len(predictions)), predictions, yerr=errors, fmt="o", label="Prediction")
        axes[idx].set_title(f"Subject: {subject_name}, RMSE: {rmse:.2f}, R2: {r2:.2f}, PCC: {pcc:.2f}")

    fig.suptitle(
        f"RMSE: {np.mean(rmse_all):.2f} +- {np.std(rmse_all):.2f}, R2: {np.mean(r2_all):.2f} +- {np.std(r2_all):.2f}, pcc: {np.mean(pcc_all):.2f} +- {np.std(pcc_all):.2f}")
    plt.legend()
    plt.savefig(file_name)
    # plt.show()
    plt.clf()
    plt.close()


def evaluate_sample_predictions_individual(value_df: pd.DataFrame, exp_name: str, dst_path: str):
    if not exists(dst_path):
        makedirs(dst_path)

    for subject_name in value_df["subject"].unique():
        subject_df = value_df[value_df["subject"] == subject_name]
        ground_truth = subject_df["ground_truth"].values
        predictions = subject_df["prediction"].values

        r2 = r2_score(ground_truth, predictions)
        mape = mean_absolute_percentage_error(predictions, ground_truth)
        mae = mean_absolute_error(predictions, ground_truth)
        rmse = mean_squared_error(predictions, ground_truth, squared=False)

        plt.figure(figsize=(column_width * cm, column_width * cm), dpi=dpi)
        plt.plot(ground_truth, label="Ground Truth")
        plt.plot(predictions, label="Prediction")
        plt.title(f"$R^2$={r2:.2f}, MAPE={mape:.2f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
        plt.xlabel("Repetition")
        plt.ylabel(y_labels[exp_name])
        plt.legend()
        plt.tight_layout()

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
