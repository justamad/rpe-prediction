from src.ml import MLOptimization, eliminate_features_with_rfe
from src.dataset import extract_dataset_input_output, normalize_subject_rpe, split_data_based_on_pseudonyms
from datetime import datetime
from sklearn.svm import SVR, SVC
from argparse import ArgumentParser
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from os.path import join, exists

import pandas as pd
import numpy as np
import logging
import os
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


def train_model(
        df: pd.DataFrame,
        task: str,
        search: str,
        n_features: int,
        log_path: str,
):
    log_path = join(log_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    if not exists(log_path):
        os.makedirs(log_path)

    X, y = extract_dataset_input_output(df)
    # y = normalize_subject_rpe(y)
    X = (X - X.mean()) / X.std()

    X, _report_df = eliminate_features_with_rfe(
        X_train=X,
        y_train=y["rpe"],
        step=25,
        n_features=n_features,
    )
    _report_df.to_csv(join(log_path, "rfe_report.csv"))
    X.to_csv(join(log_path, "X.csv"))
    y.to_csv(join(log_path, "y.csv"))

    ml_optimization = MLOptimization(
        X=X,
        y=y,
        task=task,
        mode=search,
    )
    ml_optimization.perform_grid_search_with_cv(log_path=log_path)


def evaluate_for_specific_ml_model(result_path: str, model: str, score_metric: str = "mean_test_r2"):
    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)

    result_df = pd.read_csv(join(result_path, f"{model}.csv"))
    best_combination = result_df.sort_values(by=score_metric, ascending=True).iloc[0]
    best_combination = best_combination[best_combination.index.str.contains("param")]
    param = {k.replace(f"param_{model}__", ""): v for k, v in best_combination.to_dict().items()}
    model = SVR(**param)

    steps = [
        ("balance_sampling", RandomOverSampler()),
        ("svm", model),
    ]
    pipe = Pipeline(steps=steps)

    X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X, y, train_p=0.6, random_seed=42)
    pipe.fit(X_train, y_train["rpe"])

    subjects = y_test["subject"].unique()
    fig, axes = plt.subplots(len(subjects), 1, figsize=(5, 10))
    for idx, cur_subject in enumerate(subjects):
        subject_mask = y_test["subject"] == cur_subject
        y_pred = model.predict(X_test[subject_mask])
        axes[idx].plot(y_pred, label="Predicted")
        axes[idx].plot(y_test[subject_mask]["rpe"].to_numpy(), label="Actual")
        axes[idx].set_title(f"Subject: {cur_subject}")

    plt.legend()
    plt.show()


# def aggregate_individual_ml_trials_of_model(input_path: str, ml_model: str = "svr"):
#     file_name = join(input_path, ml_model, f"{ml_model}_results.csv")
#     if isfile(file_name):
#         df = pd.read_csv(file_name, sep=";", index_col=False)
#         return df
#
#     results_data = []
#
#     for trial_file in filter(lambda x: x.endswith('csv'), os.listdir(join(input_path, ml_model))):
#         split = trial_file.split('_')
#         win_size, overlap = int(split[1]), float(split[3][:-4])
#         df = pd.read_csv(join(input_path, ml_model, trial_file),
#                          delimiter=';',
#                          index_col=False).sort_values(by='mean_test_r2', ascending=True)
#
#         plot_parallel_coordinates(
#             df.copy(),
#             color_column="mean_test_neg_mean_absolute_error",
#             title=f"Window Size: {win_size}, Overlap: {overlap}",
#             param_prefix=f"param_{ml_model}__",
#             file_name=join(input_path, ml_model, f"window_size_{win_size}_overlap_{overlap}.png")
#         )
#
#         df.insert(0, f'param_{ml_model}__win_size', win_size)
#         df.insert(1, f'param_{ml_model}__overlap', overlap)
#         results_data.append(df)
#
#     results_data = pd.concat(results_data, ignore_index=True).sort_values(by="mean_test_r2", ascending=True)
#     results_data.to_csv(file_name, sep=';', index=False)
#
#     plot_parallel_coordinates(
#         results_data.copy(),
#         color_column="mean_test_r2",
#         title=f"All parameters",
#         param_prefix=f"param_{ml_model}__",
#         file_name=join(input_path, ml_model, f"total.png"),
#     )
#
#     return results_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("my_logger").addHandler(console)

    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--result_path", type=str, dest="result_path", default="results")
    parser.add_argument("--eval_path", type=str, dest="eval_path", default="results/2023-01-25-13-35-58")
    parser.add_argument("--from_scratch", type=bool, dest="from_scratch", default=False)
    parser.add_argument("--task", type=str, dest="task", default="classification")
    parser.add_argument("--search", type=str, dest="search", default="grid")
    parser.add_argument("--n_features", type=int, dest="n_features", default=50)
    args = parser.parse_args()

    df = pd.read_csv(join(args.src_path, "segmented_features.csv"), index_col=0)

    drop_columns = ["duration", "peakSpeed", "rep_force", "rep_range"]
    df = df.drop(columns=drop_columns)

    eval_path = args.eval_path

    if args.from_scratch:
        train_model(df, args.task, search=args.search, n_features=args.n_features, log_path=args.result_path)
    else:
        eval_path = args.eval_path

    evaluate_for_specific_ml_model(eval_path, "svm", "mean_test_f1_score")
