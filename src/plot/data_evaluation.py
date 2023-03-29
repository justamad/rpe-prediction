from src.processing import get_hsv_color
from typing import Dict
from scipy.stats import pearsonr, spearmanr
from matplotlib import ticker
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

Y_AXIS_LIM_EPSILON = 0.5


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


def plot_prediction_results_for_sets(df: pd.DataFrame, file_name: str = None):
    sets = []
    rpe = []
    mean_predictions = []
    std_predictions = []

    for set in df["nr_set"].unique():
        sub_df = df[df["nr_set"] == set]
        prediction = sub_df["prediction"]
        ground_truth = sub_df["rpe"]
        sets.append(set)
        rpe.append(ground_truth.mean())
        mean_predictions.append(prediction.mean())
        std_predictions.append(prediction.std())

    pear, p = pearsonr(rpe, mean_predictions)
    r2 = r2_score(rpe, mean_predictions)

    # Create stacked error bars:
    plt.errorbar(sets, mean_predictions, std_predictions, fmt='ok', lw=1, ecolor='green', mfc='green')
    plt.scatter(sets, rpe, label="Ground Truth", c='red')
    plt.xticks(sets)
    plt.ylim(min(rpe) - Y_AXIS_LIM_EPSILON, max(rpe) + Y_AXIS_LIM_EPSILON)
    plt.xlabel("Set Nr")
    plt.ylabel("RPE value")
    plt.title(f"Correlation Pearson: {pear:.2f}, R2: {r2:.2f}")

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def plot_ml_predictions_for_frames(df: pd.DataFrame, file_name: str = None):
    predictions = df['prediction']
    ground_truth = df['rpe']

    plt.plot(ground_truth, label="Ground Truth")
    plt.plot(predictions, label="Predictions")
    plt.ylim(ground_truth.min() - Y_AXIS_LIM_EPSILON, ground_truth.max() + Y_AXIS_LIM_EPSILON)

    plt.xlabel("Frames (Windows)")
    plt.ylabel("RPE value")

    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)
    plt.title(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()
