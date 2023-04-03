from src.ml import MLOptimization, eliminate_features_with_rfe, regression_models, instantiate_best_model
from src.plot import evaluate_sample_predictions, evaluate_aggregated_predictions
from src.features import calculate_temporal_features
from typing import List, Dict, Tuple
from datetime import datetime
from argparse import ArgumentParser
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from os.path import join, exists
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr

from src.dataset import (
    normalize_gt_per_subject_mean,
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
)

import pandas as pd
import numpy as np
import itertools
import logging
import os
import yaml
import matplotlib
matplotlib.use("WebAgg")


def train_model(
        df: pd.DataFrame,
        log_path: str,
        task: str,
        normalization_input: str,
        normalization_labels: Tuple[str, bool],
        search: str,
        n_features: int,
        ground_truth: str,
        balancing: bool = False,
        temporal_features: bool = False,
        drop_columns: List = None,
        drop_prefixes: List = None,
):
    if drop_prefixes is None:
        drop_prefixes = []
    if drop_columns is None:
        drop_columns = []

    X, y = extract_dataset_input_output(df=df, labels=ground_truth)
    for prefix in drop_prefixes:
        drop_columns += [col for col in df.columns if col.startswith(prefix)]

    X.drop(columns=drop_columns, inplace=True, errors="ignore")

    if temporal_features:
        X = calculate_temporal_features(X, y, folds=2)

    if normalization_input:
        if normalization_input == "subject":
            X = normalize_data_by_subject(X, y)
        elif normalization_input == "global":
            X = normalize_data_global(X)
        else:
            raise ValueError(f"Unknown normalization_input: {normalization_input}")

    label_mean, label_std = float('inf'), float('inf')
    if normalization_labels:
        if normalization_labels == "subject":
            y = normalize_gt_per_subject_mean(y, ground_truth, "mean")
        elif normalization_labels == "global":
            values = y.loc[:, ground_truth].values
            label_mean, label_std = values.mean(), values.std()
            y.loc[:, ground_truth] = (values - label_mean) / label_std
        else:
            raise ValueError(f"Unknown normalization_labels: {normalization_labels}")

    X.fillna(0, inplace=True)
    X, _report_df = eliminate_features_with_rfe(X_train=X, y_train=y[ground_truth], step=100, n_features=n_features)
    _report_df.to_csv(join(log_path, "rfe_report.csv"))
    X.to_csv(join(log_path, "X.csv"))
    y.to_csv(join(log_path, "y.csv"))

    with open(join(log_path, "config.yml"), "w") as f:
        yaml.dump(
            {
                "task": task,
                "search": search,
                "n_features": n_features,
                "drop_columns": drop_columns,
                "ground_truth": ground_truth,
                "drop_prefixes": drop_prefixes,
                "normalization_input": normalization_input,
                "temporal_features": temporal_features,
                "balancing": balancing,
                "normalization_labels": normalization_labels,
                "label_mean": float(label_mean),
                "label_std": float(label_std),
            },
            f,
        )

    MLOptimization(
        X=X,
        y=y,
        task=task,
        mode=search,
        balance=balancing,
        ground_truth=ground_truth,
        n_groups=4,
    ).perform_grid_search_with_cv(models=regression_models, log_path=log_path)


def evaluate_entire_experiment_path(src_path: str):
    result_files = []

    for root, _, files in os.walk(src_path):
        for model_file in list(filter(lambda x: "model__" in x, files)):
            result_dict = evaluate_for_specific_ml_model(root, model_file, aggregate=True)
            result_dict["model"] = model_file.replace("model__", "").replace(".csv", "")
            result_dict["path"] = root
            result_files.append(result_dict)

    result_df = pd.DataFrame.from_dict(result_files)
    result_df.to_csv(join(src_path, "results.csv"))


def evaluate_for_specific_ml_model(result_path: str, model_file: str, aggregate: bool = False) -> Dict:
    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)

    result_df = pd.read_csv(join(result_path, model_file))
    model_name = model_file.replace("model__", "").replace(".csv", "")
    model = instantiate_best_model(result_df, model_name, config["task"])

    label_column = config["ground_truth"]

    metrics = {
        "mae": mean_absolute_percentage_error,
        "mse": mean_squared_error,
        "r2": r2_score,
        "mape": mean_absolute_percentage_error,
        "pcc": pearsonr,
    }

    test_subject_result = {metric: [] for metric in metrics.keys()}
    subjects = {}
    for idx, cur_subject in enumerate(y["subject"].unique()):
        logging.info(f"Evaluating {model_name} on subject {cur_subject}...")

        if config["balancing"]:
            model = Pipeline(steps=[
                ("balance_sampling", RandomOverSampler()),
                ("learner", model),
            ])

        X_train = X.loc[y["subject"] != cur_subject, :]
        y_train = y.loc[y["subject"] != cur_subject, :]
        X_test = X.loc[y["subject"] == cur_subject, :]
        y_test = y.loc[y["subject"] == cur_subject, :]

        ground_truth = y_test.loc[:, label_column].values
        predictions = model.fit(X_train, y_train[label_column]).predict(X_test)

        if config["normalization_labels"] == "global":
            ground_truth = ground_truth * config["label_std"] + config["label_mean"]
            predictions = predictions * config["label_std"] + config["label_mean"]
        elif config["normalization_labels"] == "subject":
            raise NotImplementedError("Subject normalization not implemented yet.")

        # Calculate all error metrics
        for name, metric in metrics.items():
            test_subject_result[name].append(metric(ground_truth, predictions))

        df = pd.DataFrame({"ground_truth": ground_truth, "predictions": predictions, "set_id": y_test["set_id"]})
        subjects[cur_subject] = df

        if aggregate:
            # raise NotImplementedError("Aggregation not implemented yet.")
            print("Aggregation not implemented yet.")

    mean_result = {f"{metric} mean": np.mean(values) for metric, values in test_subject_result.items()}
    std_result = {f"{metric} std": np.std(values) for metric, values in test_subject_result.items()}

    evaluate_sample_predictions(
        result_dict=subjects,
        gt_column="ground_truth",
        file_name=join(result_path, f"new_{model_name}_sample_prediction.png"),
    )

    if aggregate:
        evaluate_aggregated_predictions(
            result_dict=subjects,
            gt_column="ground_truth",
            file_name=join(result_path, f"new_{model_name}_aggregated_prediction.png"),
        )

    return {**mean_result, **std_result}


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
    parser.add_argument("--src_file", type=str, dest="src_file", default="data/training/statistical_features.csv")
    parser.add_argument("--result_path", type=str, dest="result_path", default="results")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments_ml")
    parser.add_argument("--train", type=bool, dest="train", default=True)
    parser.add_argument("--eval", type=bool, dest="eval", default=False)
    args = parser.parse_args()

    if args.eval:
        evaluate_entire_experiment_path("results/rpe")

    if args.train:
        df = pd.read_csv(args.src_file, index_col=0)
        experiments = list(filter(lambda x: os.path.isdir(join(args.exp_path, x)), os.listdir(args.exp_path)))
        for experiment_folder in experiments:
            exp_files = filter(lambda f: not f.startswith("_"), os.listdir(join(args.exp_path, experiment_folder)))

            for exp_name in exp_files:
                exp_config = yaml.load(open(join(args.exp_path, experiment_folder, exp_name), "r"), Loader=yaml.FullLoader)

                # Construct Search space with defined experiments
                elements = {key.replace("opt_", ""): value for key, value in exp_config.items() if key.startswith("opt_")}
                for name in elements.keys():
                    del exp_config[f"opt_{name}"]

                for combination in itertools.product(*elements.values()):
                    combination = dict(zip(elements.keys(), combination))
                    exp_config.update(combination)
                    cur_name = exp_name.replace(".yaml", "_") + "_".join([f"{key}_{value}" for key, value in combination.items()])

                    logging.info(f"Start to process experiment: {cur_name}")
                    log_path = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{cur_name}"
                    log_path = join(args.result_path, experiment_folder, log_path)
                    if not exists(log_path):
                        os.makedirs(log_path)

                    train_model(df, log_path, **exp_config)
