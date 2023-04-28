from src.ml import MLOptimization, eliminate_features_with_rfe, regression_models, instantiate_best_model
from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists

from src.plot import (
    evaluate_sample_predictions_individual,
    evaluate_nr_features,
    create_train_table,
    create_retrain_table,
    plot_subject_performance,
)

from src.dataset import (
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
    filter_labels_outliers,
    filter_outliers_z_scores,
    drop_highly_correlated_features,
    normalize_labels_min_max,
    calculate_trend_labels,
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
        normalization_labels: bool,
        search: str,
        n_features: int,
        ground_truth: str,
        n_splits: int,
        label_processing: Union[str, bool],
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

    # Impute dataframe, remove highly correlated features, and eliminate useless features
    X.fillna(0, inplace=True)
    X = drop_highly_correlated_features(X, threshold=0.95)
    X, y = filter_labels_outliers(X, y, ground_truth)
    X = filter_outliers_z_scores(X, sigma=3.0)

    label_mean, label_std = float('inf'), float('inf')
    if normalization_labels:
        values = y.loc[:, ground_truth].values
        label_mean, label_std = values.mean(), values.std()
        y.loc[:, ground_truth] = (values - label_mean) / label_std
    elif label_processing == "min_max":
        y = normalize_labels_min_max(y, ground_truth)
    elif label_processing == "trend":
        y = calculate_trend_labels(y, ground_truth)

    X, _report_df = eliminate_features_with_rfe(X_train=X, y_train=y[ground_truth], step=25, n_features=n_features)
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
                "n_splits": n_splits,
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
        n_splits=n_splits,
    ).perform_grid_search_with_cv(models=regression_models, log_path=log_path, verbose=2)


def evaluate_entire_experiment_path(
        src_path: str,
        dst_path: str,
        filter_exp: str = "",
        aggregate: bool = False,
        criteria_rank: str = "rank_test_r2",
        criteria_score: str = "mean_test_r2",
) -> pd.DataFrame:
    exp_name = os.path.basename(os.path.normpath(src_path))
    dst_path = join(dst_path, exp_name, filter_exp)
    if not exists(dst_path):
        os.makedirs(dst_path)

    result_files = []
    for root, _, files in os.walk(src_path):
        if "config.yml" not in files:
            continue

        if filter_exp not in root:
            continue

        config = yaml.load(open(join(root, "config.yml"), "r"), Loader=yaml.FullLoader)
        dp_c = ["drop_columns", "drop_prefixes", "task", "search", "ground_truth", "label_mean", "label_std"]
        for k in dp_c:
            del config[k]

        for model_file in list(filter(lambda x: "model__" in x, files)):
            model_df = pd.read_csv(join(root, model_file), index_col=0)
            best_combination = model_df.sort_values(by=criteria_rank, ascending=True).iloc[0]
            best_combination = best_combination[best_combination.index.str.contains("mean_test|std_test|rank_")]
            config["model"] = model_file.replace("model__", "").replace(".csv", "")
            config["result_path"] = root
            config["model_file"] = model_file
            result_files.append(pd.concat([best_combination, pd.Series(config)]))

    result_df = pd.DataFrame.from_records(result_files)
    result_df.to_csv(join(dst_path, "results.csv"))
    # evaluate_nr_features(result_df, dst_path)
    create_train_table(result_df, dst_path)
    logging.info("Collected all trial data. Now evaluating the best model for each ML model.")

    retrain_df = pd.DataFrame()
    for model in result_df["model"].unique():
        best_model = result_df[result_df["model"] == model].sort_values(by=criteria_score, ascending=False).iloc[0]
        df = retrain_model(best_model["result_path"], best_model["model_file"], dst_path, filter_exp)
        df["model"] = model

        if aggregate:
            df = aggregate_results(df, weighting=False)

        evaluate_sample_predictions_individual(value_df=df, exp_name=exp_name, dst_path=join(dst_path, model))
        # evaluate_aggregated_predictions(df, "rpe", join(dst_path, model))
        # plot_subject_performance(df, join(dst_path, model))
        retrain_df = pd.concat([retrain_df, df])

    final_df = create_retrain_table(retrain_df, dst_path)
    return final_df


def aggregate_results(df: pd.DataFrame, weighting: bool = False):
    result_df = pd.DataFrame()
    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        for subject_name in model_df["subject"].unique():
            subject_df = model_df[model_df["subject"] == subject_name]

            data = {"ground_truth": [], "prediction": [], "set_id": []}

            for set_id in subject_df["set_id"].unique():
                set_df = subject_df[subject_df["set_id"] == set_id]
                data["ground_truth"].append(set_df["ground_truth"].to_numpy().mean())
                if weighting:
                    data["prediction"].append(np.average(set_df["prediction"], weights=np.arange(len(set_df))))
                else:
                    data["prediction"].append(np.average(set_df["prediction"]))

                data["set_id"].append(set_id)

            temp_df = pd.DataFrame(data)
            temp_df["model"] = model
            temp_df["subject"] = subject_name
            result_df = pd.concat([result_df, temp_df])

    return result_df


def retrain_model(result_path: str, model_file: str, dst_path: str, filter_exp: str) -> pd.DataFrame:
    model_name = model_file.replace("model__", "").replace(".csv", "")
    result_filename = join(dst_path, f"{model_name}_{filter_exp + '_' if filter_exp else ''}results.csv")
    if exists(result_filename):
        logging.info(f"Skip re-training of {model_name.upper()} as result already exists.")
        return pd.read_csv(result_filename, index_col=0)

    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    res_df = pd.read_csv(join(result_path, model_file))
    model = instantiate_best_model(res_df, model_name, config["task"])

    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)

    logging.info(f"Evaluate {model_name.upper()} model from path {result_path}")
    opt = MLOptimization(
        X=X,
        y=y,
        task=config["task"],
        mode=config["search"],
        balance=config["balancing"],
        ground_truth=config["ground_truth"],
        n_splits=config["n_splits"],
    )
    res_df = opt.evaluate_model(model, config["normalization_labels"], config["label_mean"], config["label_std"])
    res_df.to_csv(result_filename)
    return res_df


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
    parser.add_argument("--train", type=bool, dest="train", default=True)
    parser.add_argument("--eval", type=bool, dest="eval", default=False)
    args = parser.parse_args()

    if args.train:
        experiments = list(filter(lambda x: os.path.isdir(join(args.exp_path, x)), os.listdir(args.exp_path)))
        for experiment_folder in experiments:
            exp_files = filter(lambda f: not f.startswith("_"), os.listdir(join(args.exp_path, experiment_folder)))

            for exp_name in exp_files:
                exp_cfg = yaml.load(open(join(args.exp_path, experiment_folder, exp_name), "r"), Loader=yaml.FullLoader)

                # Load data
                file_names = exp_cfg["training_file"]
                if isinstance(file_names, str):
                    file_names = [file_names]

                df = pd.read_csv(join(args.src_path, file_names[0]), index_col=0)
                for file_name in file_names[1:]:
                    add_df = pd.read_csv(join(args.src_path, file_name), index_col=0)
                    add_df.drop([c for c in df.columns if c in add_df.columns], axis=1, inplace=True)
                    df = pd.concat([df, add_df], axis=1)

                del exp_cfg["training_file"]

                # Construct search space with defined experiments
                elements = {key.replace("opt_", ""): value for key, value in exp_cfg.items() if key.startswith("opt_")}
                for name in elements.keys():
                    del exp_cfg[f"opt_{name}"]

                for combination in itertools.product(*elements.values()):
                    combination = dict(zip(elements.keys(), combination))
                    exp_cfg.update(combination)
                    cur_name = exp_name.replace(".yaml", "_") + "_".join(
                        [f"{key}_{value}" for key, value in combination.items()])

                    logging.info(f"Start to process experiment: {cur_name}")
                    log_path = join(args.result_path, experiment_folder,
                                    f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{cur_name}")
                    if not exists(log_path):
                        os.makedirs(log_path)

                    train_model(df, log_path, **exp_cfg)

    if args.eval:

        def merge_experiments(exp_name: str, aggregate: bool):
            full_df = evaluate_entire_experiment_path(exp_name, args.dst_path, "full", aggregate)
            full_df.columns = [(c, "Full Rep") for c in full_df.columns]
            ecc_con_df = evaluate_entire_experiment_path(exp_name, args.dst_path, "con_ecc", aggregate)
            ecc_con_df.columns = [(c, "Con / Ecc") for c in ecc_con_df.columns]
            merge_df = pd.concat([full_df, ecc_con_df], axis=1)
            merge_df.columns = pd.MultiIndex.from_tuples(merge_df.columns, names=['Model', 'Mode'])
            merge_df.sort_index(axis=1, level=[0, 1], ascending=[True, True], inplace=True)
            merge_df.to_latex(
                f"{exp_name.replace('/', '_')}.txt", escape=False,
                column_format="l" + "r" * (len(merge_df.columns))
            )

        # merge_experiments("results/rpe", aggregate=False)
        # con_df = evaluate_entire_experiment_path("results/powercon", args.dst_path, aggregate=False)
        # con_df.columns = [(c, "Con") for c in con_df.columns]
        # ecc_df = evaluate_entire_experiment_path("results/powerecc", args.dst_path, aggregate=False)
        # ecc_df.columns = [(c, "Ecc") for c in ecc_df.columns]
        # merge_df = pd.concat([con_df, ecc_df], axis=1)
        # merge_df.columns = pd.MultiIndex.from_tuples(merge_df.columns, names=['Model', 'Mode'])
        # merge_df.sort_index(axis=1, level=[0, 1], ascending=[True, True], inplace=True)
        # merge_df.to_latex(
        #     f"power.txt", escape=False,
        #     column_format="l" + "r" * (len(merge_df.columns))
        # )
        # evaluate_entire_experiment_path("results/rpe_scaled", args.dst_path, aggregate=True)
        merge_experiments("results/rpe_scaled", aggregate=False)
