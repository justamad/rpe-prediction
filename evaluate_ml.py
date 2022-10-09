from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from imblearn.pipeline import Pipeline
from scipy.stats import spearmanr, pearsonr
from src.utils import split_data_based_on_pseudonyms
from os.path import join, isfile
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

from src.plot import (
    plot_parallel_coordinates,
    plot_prediction_results_for_sets,
    plot_ml_predictions_for_frames,
)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_path", type=str, dest="src_path", default="results/2022-07-25-11-27-00_flywheel_mean")
parser.add_argument("--model", type=str, dest="model", default="svr")
args = parser.parse_args()


def evaluate_best_performing_ml_model(input_path: str, ml_model: str, score_metric: str):
    df = pd.read_csv(join(input_path, f"{ml_model}.csv"), sep=";", index_col=0)
    df = df[df["rank_test_accuracy_score"] == 1]
    i = 12

    # best_combination = df.sort_values(by=score_metric, ascending=True).iloc[0]
    # best_combination = best_combination[best_combination.index.str.contains("param")]
    # param = {k.replace(f"param_{ml_model}__", ""): v for k, v in best_combination.to_dict().items()}
    # model = SVC(**param)
    # df = pd.read_csv(join(input_path, "X_rfe.csv"), sep=";", index_col=False)
    # train_general_model(df, model, score_metric)
    # personalized_model(df, model, score_metric)


def train_general_model(df: pd.DataFrame, model, score_metric: str):
    X_train = df.iloc[:, :-4]
    y_train = df.iloc[:, -4:]

    X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X_train, y_train, train_p=0.5, random_seed=17)
    y_train = y_train["rpe"]
    y_test = y_test["rpe"]

    steps = [
        ("balance_sampling", RandomOverSampler()),
        ("svm", model),
    ]
    pipe = Pipeline(steps=steps)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print(f"{model} - {score_metric}: {accuracy_score(y_test, y_pred)}")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)

    agg = np.column_stack([y_pred, y_test])
    agg = pd.DataFrame(agg, columns=["pred", "true"])
    agg = agg.sort_values(by="true", ascending=True)

    fig, axes = plt.subplots(2, figsize=(5, 10))
    plt.title(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    axes[0].plot(agg["pred"].to_numpy(), label="Predicted")
    axes[0].plot(agg["true"].to_numpy(), label="Actual")
    sns.heatmap(matrix / np.sum(matrix), ax=axes[1], annot=True, fmt=".2%", cmap="Blues")
    plt.legend()
    plt.show()


def personalized_model(df: pd.DataFrame, model, score_metric: str):
    for subject in df.subject.unique():
        sub_df = df[df["subject"] == subject]
        X_train = sub_df.iloc[:, :-4]
        y_train = sub_df["rpe"]

        # X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X_train, y_train, train_p=0.7, random_seed=42)
        # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=17)
        split = StratifiedKFold(n_splits=20, shuffle=True, random_state=17)
        s = split.split(X_train, y_train)

        # model.fit(X_train, y_train)

        steps = [
            ("balance_sampling", RandomOverSampler()),
            ("svm", model),
        ]
        pipe = Pipeline(steps=steps)
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            print(e)
            continue

        y_pred = pipe.predict(X_test)
        matrix = confusion_matrix(y_test, y_pred)

        agg = np.column_stack([y_pred, y_test])
        agg = pd.DataFrame(agg, columns=["pred", "true"])
        agg = agg.sort_values(by="true", ascending=True)

        fig, axes = plt.subplots(2, figsize=(5, 10))
        plt.title(f"{subject}, Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        axes[0].plot(agg["pred"].to_numpy(), label="Predicted")
        axes[0].plot(agg["true"].to_numpy(), label="Actual")
        sns.heatmap(matrix / np.sum(matrix), ax=axes[1], annot=True, fmt=".2%", cmap="Blues")
        plt.legend()
        plt.show()


def aggregate_individual_ml_trials_of_model(input_path: str, ml_model: str = "svr"):
    file_name = join(input_path, ml_model, f"{ml_model}_results.csv")
    if isfile(file_name):
        df = pd.read_csv(file_name, sep=";", index_col=False)
        return df

    results_data = []

    for trial_file in filter(lambda x: x.endswith('csv'), os.listdir(join(input_path, ml_model))):
        split = trial_file.split('_')
        win_size, overlap = int(split[1]), float(split[3][:-4])
        df = pd.read_csv(join(input_path, ml_model, trial_file),
                         delimiter=';',
                         index_col=False).sort_values(by='mean_test_r2', ascending=True)

        plot_parallel_coordinates(
            df.copy(),
            color_column="mean_test_neg_mean_absolute_error",
            title=f"Window Size: {win_size}, Overlap: {overlap}",
            param_prefix=f"param_{ml_model}__",
            file_name=join(input_path, ml_model, f"window_size_{win_size}_overlap_{overlap}.png")
        )

        df.insert(0, f'param_{ml_model}__win_size', win_size)
        df.insert(1, f'param_{ml_model}__overlap', overlap)
        results_data.append(df)

    results_data = pd.concat(results_data, ignore_index=True).sort_values(by="mean_test_r2", ascending=True)
    results_data.to_csv(file_name, sep=';', index=False)

    plot_parallel_coordinates(
        results_data.copy(),
        color_column="mean_test_r2",
        title=f"All parameters",
        param_prefix=f"param_{ml_model}__",
        file_name=join(input_path, ml_model, f"total.png"),
    )

    return results_data


if __name__ == '__main__':
    evaluate_best_performing_ml_model(args.src_path, args.model, "rank_test_accuracy_score")
