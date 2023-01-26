from typing import List
from src.ml import MLOptimization, eliminate_features_with_rfe
from datetime import datetime
from sklearn.svm import SVR, SVC
from argparse import ArgumentParser
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from os.path import join, exists
from sklearn.metrics import mean_squared_error
from src.dataset import (
    extract_dataset_input_output,
    normalize_subject_rpe,
    split_data_based_on_pseudonyms,
    normalize_data_by_subject,
)

import pandas as pd
import numpy as np
import logging
import os
import yaml
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def train_model(
        df: pd.DataFrame,
        exp_name: str,
        log_path: str,
        task: str,
        search: str,
        n_features: int,
        ground_truth: str,
        drop_columns=None,
        drop_prefixes=None,

):
    if drop_prefixes is None:
        drop_prefixes = []
    if drop_columns is None:
        drop_columns = []

    log_path = join(log_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{exp_name}")
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
                "drop_prefixes": drop_prefixes,
            },
            f,
        )

    X, y = extract_dataset_input_output(df=df, ground_truth_column=ground_truth)

    for prefix in drop_prefixes:
        drop_columns += [col for col in df.columns if col.startswith(prefix)]

    X.drop(columns=drop_columns, inplace=True, errors="ignore")

    # TODO: Subject dependent normalization
    # y = normalize_subject_rpe(y)
    X = normalize_data_by_subject(X, y)
    X.fillna(0, inplace=True)
    # X = (X - X.mean()) / X.std()

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
    return log_path


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

    X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X, y, train_p=0.6, random_seed=17)
    gt_column = config["ground_truth"]
    model.fit(X_train, y_train[gt_column])

    y_test["pred"] = model.predict(X_test)
    subjects = y_test["subject"].unique()
    fig, axes = plt.subplots(len(subjects), 1, sharey=True, figsize=(10, 10))

    rmse = mean_squared_error(y_test[gt_column], y_test["pred"], squared=False)
    for idx, cur_subject in enumerate(subjects):
        ground_truth = y_test.loc[y_test["subject"] == cur_subject, gt_column].to_numpy()
        predictions = y_test.loc[y_test["subject"] == cur_subject, "pred"].to_numpy()

        axes[idx].plot(ground_truth, label="Ground Truth")
        axes[idx].plot(predictions, label="Prediction")
        axes[idx].set_title(f"Subject: {cur_subject}")

    fig.suptitle(f"RMSE: {rmse:.2f}", fontsize=30)
    plt.legend()
    plt.savefig(join(result_path, "eval.png"))
    # plt.show()
    plt.clf()
    plt.close()


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
    parser.add_argument("--eval_path", type=str, dest="eval_path", default="results/2023-01-26-11-02-24_no_hrv")
    parser.add_argument("--from_scratch", type=bool, dest="from_scratch", default=True)
    parser.add_argument("--task", type=str, dest="task", default="classification")
    parser.add_argument("--search", type=str, dest="search", default="grid")
    parser.add_argument("--n_features", type=int, dest="n_features", default=100)
    args = parser.parse_args()

    exp_path = "experiments"
    df = pd.read_csv(join(args.src_path, "seg_hrv.csv"), index_col=0)

    # if args.from_scratch:
    for experiment in os.listdir(exp_path):
        exp_config = yaml.load(open(join(exp_path, experiment), "r"), Loader=yaml.FullLoader)
        eval_path = train_model(df, experiment.replace(".yaml", ""), args.result_path, **exp_config)
        evaluate_for_specific_ml_model(eval_path, "svm", "mean_test_f1_score")

        # eval_path = train_model(
        #     df=df,
        #     task=args.task,
        #     search=args.search,
        #     n_features=args.n_features,
        #     # drop_columns=["HRV_Load (TRIMP)", "powerEcc", "duration", "peakSpeed", "rep_force", "rep_range", "powerAvg"],
        #     # drop_columns=["HRV_Load (TRIMP)", "powerEcc", "duration", "peakSpeed", "rep_force", "rep_range", "powerAvg"],
        #     # drop_columns=["HRV_"],
        #     ground_truth="rpe",
        #     # drop_prefixes=["MOCAP", "HRV"],
        #     log_path=args.result_path,
        # )
    #     pass
    # else:
    #     eval_path = args.eval_path

