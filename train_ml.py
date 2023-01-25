from typing import List
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
import yaml
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
        drop_columns: List[str],
        ground_truth: str,
        log_path: str,
):
    log_path = join(log_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    if not exists(log_path):
        os.makedirs(log_path)

    with open(join(log_path, "config.yml"), "w") as f:
        yaml.dump(
            {
                "task": task,
                "search": search,
                "n_features": n_features,
                "drop_columns": drop_columns,
                "ground_truth": ground_truth,
            },
            f,
        )

    df.drop(columns=drop_columns, inplace=True, errors="ignore")
    X, y = extract_dataset_input_output(df, ground_truth)
    # y = normalize_subject_rpe(y)
    X = (X - X.mean()) / X.std()

    X, _report_df = eliminate_features_with_rfe(
        X_train=X,
        y_train=y[ground_truth],
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
        ground_truth=ground_truth,
    )
    ml_optimization.perform_grid_search_with_cv(log_path=log_path)


def evaluate_for_specific_ml_model(result_path: str, model: str, score_metric: str = "mean_test_r2"):
    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)

    result_df = pd.read_csv(join(result_path, f"{model}.csv"))
    best_combination = result_df.sort_values(by=score_metric, ascending=True).iloc[0]
    best_combination = best_combination[best_combination.index.str.contains("param")]
    param = {k.replace(f"param_{model}__", ""): v for k, v in best_combination.to_dict().items()}
    model = SVR(**param)

    if config["task"] == "classification":
        model = Pipeline(steps=[
            ("balance_sampling", RandomOverSampler()),
            ("svm", model),
        ])

    X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X, y, train_p=0.6, random_seed=42)
    ground_truth = config["ground_truth"]
    model.fit(X_train, y_train[ground_truth])

    subjects = y_test["subject"].unique()
    fig, axes = plt.subplots(len(subjects), 1, figsize=(5, 10))
    for idx, cur_subject in enumerate(subjects):
        subject_mask = y_test["subject"] == cur_subject
        y_pred = model.predict(X_test[subject_mask])
        axes[idx].plot(y_pred, label="Predicted")
        axes[idx].plot(y_test[subject_mask][ground_truth].to_numpy(), label="Actual")
        axes[idx].set_title(f"Subject: {cur_subject}")

    plt.legend()
    plt.show()


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
    parser.add_argument("--eval_path", type=str, dest="eval_path", default="results/2023-01-25-14-37-06")
    parser.add_argument("--from_scratch", type=bool, dest="from_scratch", default=False)
    parser.add_argument("--task", type=str, dest="task", default="regression")
    parser.add_argument("--search", type=str, dest="search", default="grid")
    parser.add_argument("--n_features", type=int, dest="n_features", default=50)
    args = parser.parse_args()

    df = pd.read_csv(join(args.src_path, "seg_hrv.csv"), index_col=0)
    eval_path = args.eval_path

    if args.from_scratch:
        train_model(
            df=df,
            task=args.task,
            search=args.search,
            n_features=args.n_features,
            drop_columns=["powerCon", "powerEcc", "duration", "peakSpeed", "rep_force", "rep_range", "powerAvg"],
            ground_truth="rpe",
            log_path=args.result_path,
        )
    else:
        eval_path = args.eval_path

    evaluate_for_specific_ml_model(eval_path, "svr", "mean_test_r2")
