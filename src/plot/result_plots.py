from .plot_settings import column_width, text_width, cm, dpi
from typing import Dict
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
import pandas as pd
import numpy as np


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


def evaluate_sample_predictions_individual(result_dict: Dict, gt_column: str, dst_path: str):
    rmse_all = []
    r2_all = []
    mape_all = []

    if not exists(dst_path):
        makedirs(dst_path)

    for idx, (subject_name, df) in enumerate(result_dict.items()):
        plt.figure(figsize=(column_width * cm, column_width * cm), dpi=dpi)

        ground_truth = df[gt_column].to_numpy()
        predictions = df["predictions"].to_numpy()
        rmse = mean_squared_error(predictions, ground_truth, squared=False)
        r2 = r2_score(ground_truth, predictions)
        rmse_all.append(rmse)
        r2_all.append(r2)
        mape = mean_absolute_percentage_error(predictions, ground_truth)
        mape_all.append(mape)

        plt.plot(ground_truth, label="Ground Truth")
        plt.plot(predictions, label="Prediction")
        plt.title(f"RMSE: {rmse:.2f}, R2: {r2:.2f}, MAPE: {mape:.2f}")
        plt.legend()

        plt.savefig(join(dst_path, f"{subject_name}.png"))
        # plt.show()
        plt.clf()
        plt.close()
