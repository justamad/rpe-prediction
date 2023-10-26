import pandas as pd
import itertools
import logging
import os
import yaml
import matplotlib

from src.ml import MLOptimization, eliminate_features_with_rfe, regression_models, instantiate_best_model
from src.dataset import aggregate_results
from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists

from src.plot import (
    plot_sample_predictions,
    create_train_table,
    create_retrain_table,
    plot_subject_correlations,
    create_bland_altman_plot,
    create_scatter_plot,
    create_residual_plot,
)

from src.dataset import (
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
    filter_labels_outliers_per_subject,
    clip_outliers_z_scores,
    drop_highly_correlated_features,
    add_rolling_statistics,
)


def train_models_with_grid_search(
        df: pd.DataFrame,
        log_path: str,
        task: str,
        normalization_input: str,
        normalization_labels: bool,
        search: str,
        n_features: int,
        ground_truth: str,
        n_splits: int,
        rolling_statistics: Union[int, bool],
        balancing: bool = False,
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
    X, y = filter_labels_outliers_per_subject(X, y, ground_truth, sigma=3.0)
    X = clip_outliers_z_scores(X, sigma=3.0)

    label_mean, label_std = float("inf"), float("inf")
    if normalization_labels:
        values = y.loc[:, ground_truth].values
        label_mean, label_std = values.mean(), values.std()
        y.loc[:, ground_truth] = (values - label_mean) / label_std

    X, _report_df = eliminate_features_with_rfe(X_train=X, y_train=y[ground_truth], step=25, n_features=n_features)
    _report_df.to_csv(join(log_path, "rfe_report.csv"))

    if rolling_statistics:
        X = add_rolling_statistics(X, y, win=rolling_statistics, normalize=True)

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
                "rolling_statistics": rolling_statistics,
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
        labels=ground_truth,
        n_splits=n_splits,
    ).perform_grid_search_with_cv(models=regression_models, log_path=log_path, verbose=2)


def evaluate_entire_training_folder(
        src_path: str,
        dst_path: str,
        filter_exp: str = "",
        aggregate: bool = False,
) -> pd.DataFrame:
    basename = os.path.basename(src_path)

    for experiment in os.listdir(src_path):
        prediction_goal, experiment_name = experiment.split("_")
        dst_path = join(dst_path, basename, experiment_name)
        os.makedirs(dst_path, exist_ok=True)
        evaluate_experiment_path(src_path, dst_path, prediction_goal, filter_exp, aggregate)


def evaluate_experiment_path(
        src_path: str,
        dst_path: str,
        exp_name: str,
        filter_exp: str = "",
        aggregate: bool = True,
        criteria_rank: str = "rank_test_r2",
        criteria_score: str = "mean_test_r2",
) -> pd.DataFrame:
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

        plot_sample_predictions(value_df=df, exp_name=exp_name, dst_path=join(dst_path, model))

        if aggregate:
            df = aggregate_results(df)

        plot_subject_correlations(df, join(dst_path, model))
        create_bland_altman_plot(df, join(dst_path), model, exp_name)
        create_scatter_plot(df, dst_path, model, exp_name)
        create_residual_plot(df, dst_path, model)

        retrain_df = pd.concat([retrain_df, df])

    final_df = create_retrain_table(retrain_df, dst_path)
    final_df.to_csv(join(dst_path, "retrain_results.csv"))
    return final_df


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

    logging.info(f"Re-train {model_name.upper()} model from path {result_path}")
    opt = MLOptimization(
        X=X,
        y=y,
        task=config["task"],
        mode=config["search"],
        balance=config["balancing"],
        labels=config["ground_truth"],
        n_splits=config["n_splits"],
    )
    res_df = opt.evaluate_model(model, config["normalization_labels"], config["label_mean"], config["label_std"])
    res_df.to_csv(result_filename)
    return res_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, dest="data_path", default="data/training")
    parser.add_argument("--result_path", type=str, dest="result_path", default="results/ml/train")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments/ml")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="results/ml/test")
    parser.add_argument("--exp_folder", type=str, dest="exp_folder", default="results/ml/train/2023-10-26-08-20-15")
    parser.add_argument("--train", type=bool, dest="train", default=False)
    parser.add_argument("--eval", type=bool, dest="eval", default=False)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("my_logger").addHandler(console)
    matplotlib.use("WebAgg")

    if args.train:
        experiment_time = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

        exp_files = filter(lambda f: not f.startswith("_"), os.listdir(join(args.exp_path)))
        for exp_file in exp_files:
            exp_name = exp_file.replace(".yaml", "")
            config = yaml.load(open(join(args.exp_path, exp_file), "r"), Loader=yaml.FullLoader)
            file_names = config["training_file"]
            if isinstance(file_names, str):
                file_names = [file_names]

            df = pd.read_csv(join(args.data_path, file_names[0]), index_col=0)
            for file_name in file_names[1:]:
                add_df = pd.read_csv(join(args.data_path, file_name), index_col=0)
                add_df.drop([c for c in df.columns if c in add_df.columns], axis=1, inplace=True)
                df = pd.concat([df, add_df], axis=1)

            del config["training_file"]
            elements = {key.replace("opt_", ""): value for key, value in config.items() if key.startswith("opt_")}
            for name in elements.keys():
                del config[f"opt_{name}"]

            for combination in itertools.product(*elements.values()):
                combination = dict(zip(elements.keys(), combination))
                config.update(combination)

                experiment_folder = "_".join([f"{k}_{v}" for k, v in combination.items()])

                logging.info(f"Start to process experiment: {experiment_folder}")
                log_path = join(args.result_path, experiment_time, exp_name, experiment_folder)
                if not exists(log_path):
                    os.makedirs(log_path)

                train_models_with_grid_search(df, log_path, **config)

    if args.eval:
        evaluate_entire_training_folder(args.exp_folder, args.dst_path, "", aggregate=True)

        # evaluate_entire_experiment_path("data/ml_results/poweravg", args.dst_path, "", aggregate=False)
        #
        # t_ml = pd.read_csv("data/ml_evaluation/rpe/retrain_results.csv", index_col=0)
        # d_ml = pd.read_csv("data/dl_evaluation/rpe/retrain.csv", index_col=0)
        # total_df = pd.concat([t_ml, d_ml], axis=1)
        # total_df.to_latex("rpe.tex", column_format="l" + "r" * (len(total_df.columns)), escape=False)
        #
        # t_ml = pd.read_csv("data/ml_evaluation/poweravg/retrain_results.csv", index_col=0)
        # d_ml = pd.read_csv("data/dl_evaluation/power/retrain.csv", index_col=0)
        # p_ml = pd.read_csv("data/physics/results.csv", index_col=0)
        # total_df = pd.concat([t_ml, d_ml, p_ml], axis=1)
        # total_df.to_latex("power.tex", column_format="l" + "r" * (len(total_df.columns)), escape=False)