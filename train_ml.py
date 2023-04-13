from src.ml import MLOptimization, eliminate_features_with_rfe, regression_models, instantiate_best_model
from typing import List, Dict, Tuple
from datetime import datetime
from argparse import ArgumentParser
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from os.path import join, exists
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr

from src.plot import (
    evaluate_sample_predictions,
    evaluate_aggregated_predictions,
    evaluate_sample_predictions_individual,
    evaluate_nr_features,
)

from src.dataset import (
    normalize_gt_per_subject_mean,
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
    filter_outliers_z_scores,
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
        drop_columns += [col for col in df.columns if prefix in col]

    X.drop(columns=drop_columns, inplace=True, errors="ignore")
    X = X.loc[:, (X != 0).any(axis=0)]

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

    # Impute data frame and eliminate features
    X.fillna(0, inplace=True)
    X = filter_outliers_z_scores(X, sigma=3.0)
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


def evaluate_entire_experiment_path(
        src_path: str,
        dst_path: str,
        criteria_rank: str = "rank_test_r2",
        criteria_score: str = "mean_test_r2",
):
    dst_path = join(dst_path, os.path.basename(os.path.normpath(src_path)))
    if not exists(dst_path):
        os.makedirs(dst_path)

    result_files = []
    for root, _, files in os.walk(src_path):
        if "config.yml" not in files:
            continue

        config = yaml.load(open(join(root, "config.yml"), "r"), Loader=yaml.FullLoader)
        dp_c = ["drop_columns", "drop_prefixes", "task", "search", "ground_truth", "label_mean", "label_std"]
        for k in dp_c:
            del config[k]

        for model_file in list(filter(lambda x: "model__" in x, files)):
            model_df = pd.read_csv(join(root, model_file), index_col=0)
            best_combination = model_df.sort_values(by=criteria_rank, ascending=True).iloc[0]
            best_combination = best_combination[best_combination.index.str.contains("mean_test|std_test|rank_")]
            config["model"] = model_file.replace("model__", "").replace(".csv", "").title()
            config["result_path"] = root
            config["model_file"] = model_file
            result_files.append(pd.concat([best_combination, pd.Series(config)]))

    result_df = pd.DataFrame.from_records(result_files)
    result_df.to_csv(join(src_path, "results.csv"))
    evaluate_nr_features(result_df, dst_path)
    return
    logging.info("Collected all trial data. Now evaluating the best model for each ML model.")

    for model in result_df["model"].unique():
        cur_df = result_df[result_df["model"] == model].sort_values(by=criteria_score, ascending=False)
        best_model = cur_df.iloc[0]
        m = evaluate_for_specific_ml_model(best_model["result_path"], best_model["model_file"], "output", aggregate=True)
        i = 12


def evaluate_for_specific_ml_model(result_path: str, model_file: str, dst_path: str, aggregate: bool = False) -> Dict:
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
    for idx, test_subject in enumerate(y["subject"].unique()):
        logging.info(f"Evaluating {model_name} on subject {test_subject}...")

        if config["balancing"]:
            model = Pipeline(steps=[
                ("balance_sampling", RandomOverSampler()),
                ("learner", model),
            ])

        X_train = X.loc[y["subject"] != test_subject, :]
        y_train = y.loc[y["subject"] != test_subject, :]
        X_test = X.loc[y["subject"] == test_subject, :]
        y_test = y.loc[y["subject"] == test_subject, :]

        ground_truth = y_test.loc[:, label_column].values
        predictions = model.fit(X_train, y_train[label_column]).predict(X_test)

        if config["normalization_labels"] == "global":
            ground_truth = ground_truth * config["label_std"] + config["label_mean"]
            predictions = predictions * config["label_std"] + config["label_mean"]

        # Calculate all error metrics
        for metric_name, metric in metrics.items():
            test_subject_result[metric_name].append(metric(ground_truth, predictions))

        df = pd.DataFrame({"ground_truth": ground_truth, "predictions": predictions, "set_id": y_test["set_id"]})
        subjects[test_subject] = df

        if aggregate:
            logging.error("Aggregation not implemented yet.")

    mean_result = {f"{metric} mean": np.mean(values) for metric, values in test_subject_result.items()}
    std_result = {f"{metric} std": np.std(values) for metric, values in test_subject_result.items()}

    # evaluate_sample_predictions(
    #     result_dict=subjects,
    #     gt_column="ground_truth",
    #     file_name=join(result_path, f"new_{model_name}_sample_prediction.png"),
    # )

    evaluate_sample_predictions_individual(
        result_dict=subjects,
        gt_column="ground_truth",
        dst_path=join(dst_path, model_name),
    )

    # if aggregate:
    #     evaluate_aggregated_predictions(
    #         result_dict=subjects,
    #         gt_column="ground_truth",
    #         file_name=join(result_path, f"new_{model_name}_aggregated_prediction.png"),
    #     )

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
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--result_path", type=str, dest="result_path", default="results")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments_ml")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="evaluation")
    parser.add_argument("--train", type=bool, dest="train", default=False)
    parser.add_argument("--eval", type=bool, dest="eval", default=True)
    args = parser.parse_args()

    if args.train:
        experiments = list(filter(lambda x: os.path.isdir(join(args.exp_path, x)), os.listdir(args.exp_path)))
        for experiment_folder in experiments:
            exp_files = filter(lambda f: not f.startswith("_"), os.listdir(join(args.exp_path, experiment_folder)))

            for exp_name in exp_files:
                exp_config = yaml.load(open(join(args.exp_path, experiment_folder, exp_name), "r"), Loader=yaml.FullLoader)

                # Load data
                file_names = exp_config["training_file"]
                if isinstance(file_names, str):
                    file_names = [file_names]

                df = pd.read_csv(join(args.src_path, file_names[0]), index_col=0)
                for file_name in file_names[1:]:
                    add_df = pd.read_csv(join(args.src_path, file_name), index_col=0)
                    add_df.drop([c for c in df.columns if c in add_df.columns], axis=1, inplace=True)
                    df = pd.concat([df, add_df], axis=1)

                del exp_config["training_file"]

                # Construct search space with defined experiments
                elements = {key.replace("opt_", ""): value for key, value in exp_config.items() if key.startswith("opt_")}
                for name in elements.keys():
                    del exp_config[f"opt_{name}"]

                for combination in itertools.product(*elements.values()):
                    combination = dict(zip(elements.keys(), combination))
                    exp_config.update(combination)
                    cur_name = exp_name.replace(".yaml", "_") + "_".join([f"{key}_{value}" for key, value in combination.items()])

                    logging.info(f"Start to process experiment: {cur_name}")
                    log_path = join(args.result_path, experiment_folder, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{cur_name}")
                    if not exists(log_path):
                        os.makedirs(log_path)

                    train_model(df, log_path, **exp_config)

    if args.eval:
        evaluate_entire_experiment_path("results/hr", args.dst_path)
