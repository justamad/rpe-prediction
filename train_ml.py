import pandas as pd
import itertools
import logging
import os
import yaml
import matplotlib

from typing import List, Union
from datetime import datetime
from argparse import ArgumentParser
from os.path import join, exists, basename
from src.ml import MLOptimization, regression_models, instantiate_best_model, eliminate_features_rfecv

from src.plot import (
    plot_feature_elimination,
    plot_sample_predictions,
    create_train_table,
    create_retrain_table,
    plot_subject_correlations,
    create_bland_altman_plot,
    create_scatter_plot,
    create_residual_plot,
    create_model_performance_plot,
    create_total_run_table,
)

from src.dataset import (
    aggregate_results,
    extract_dataset_input_output,
    normalize_data_by_subject,
    normalize_data_global,
    clip_outliers_z_scores,
    drop_correlated_features,
    add_rolling_statistics,
    remove_low_variance_features,
    get_highest_correlation_features,
)


def train_models_with_grid_search(
        df: pd.DataFrame,
        log_path: str,
        task: str,
        normalization_input: str,
        normalization_labels: bool,
        search: str,
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

    if rolling_statistics:
        X = add_rolling_statistics(X, y, win=[rolling_statistics])

    X.fillna(0, inplace=True)
    X = remove_low_variance_features(X, threshold=0.01)
    X = drop_correlated_features(X, threshold=0.95)
    X = clip_outliers_z_scores(X, sigma=3.0)
    X = get_highest_correlation_features(X, y[ground_truth], k=200)
    X = eliminate_features_rfecv(X, y, gt=ground_truth, n_splits=n_splits, steps=5, min_features=5, log_path=log_path)

    label_mean, label_std = float("inf"), float("inf")
    if normalization_labels:
        values = y.loc[:, ground_truth].values
        label_mean, label_std = values.mean(), values.std()
        y.loc[:, ground_truth] = (values - label_mean) / label_std

    X.to_csv(join(log_path, "X.csv"))
    y.to_csv(join(log_path, "y.csv"))

    with open(join(log_path, "config.yml"), "w") as f:
        yaml.dump(
            {
                "task": task,
                "search": search,
                "n_features": X.shape[1],
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
    ).perform_grid_search_with_cv(models=regression_models, log_path=log_path, verbose=2, n_jobs=4)


def evaluate_entire_training_folder(src_path: str, aggregate: bool):
    for experiment in os.listdir(src_path):
        prediction_goal, experiment_name = experiment.split("_")

        # Re-train models and perform all evaluations
        for root, _, files in os.walk(join(src_path, experiment)):
            if "config.yml" in files:
                evaluate_experiment_path(
                    src_path=root,
                    dst_path=root.replace("train", "test"),
                    exp_name=prediction_goal,
                    files=files,
                    aggregate=aggregate,
                )

        # Collect all results and create a table
        dst_path = join(src_path.replace("train", "test"), experiment)
        result_df = collect_retrain_results(dst_path, "retrain_results.csv")
        create_total_run_table(result_df, dst_path)

        # for metric in ["MSE", "RMSE", "MAE", "$R^{2}$", "MAPE"]:
        #     create_model_performance_plot(result_df, dst_path, experiment_name, metric)
        # create_model_performance_plot(result_df, dst_path, experiment_name, "Spearman's $\\rho$", alt_name="Spearman")


def evaluate_experiment_path(
        src_path: str,
        dst_path: str,
        exp_name: str,
        files: List[str] = None,
        aggregate: bool = True,
        criteria_rank: str = "rank_test_r2",
        criteria_score: str = "mean_test_r2",
) -> pd.DataFrame:
    os.makedirs(dst_path, exist_ok=True)

    result_df = collect_model_run_files(src_path, criteria_rank, files)
    result_df.to_csv(join(dst_path, "results.csv"))
    create_train_table(result_df, dst_path)
    logging.info("Collected all trial data. Now evaluating the best combination of each model.")

    # Investigate features
    rfe_df = pd.read_csv(join(src_path, "cv_results.csv"), index_col=0)
    plot_feature_elimination(rfe_df, dst_path)

    retrain_df = pd.DataFrame()
    for model in result_df["model"].unique():
        best_model = result_df[result_df["model"] == model].sort_values(by=criteria_score, ascending=False).iloc[0]
        prediction_df = retrain_model(best_model["result_path"], best_model["model_file"], dst_path)
        prediction_df["model"] = model

        plot_sample_predictions(value_df=prediction_df, exp_name=exp_name, dst_path=join(dst_path, model))

        if aggregate:
            prediction_df = aggregate_results(prediction_df)

        plot_subject_correlations(prediction_df, join(dst_path, model))
        create_bland_altman_plot(prediction_df, join(dst_path), model, exp_name)
        create_scatter_plot(prediction_df, dst_path, model, exp_name)
        create_residual_plot(prediction_df, dst_path, model)

        retrain_df = pd.concat([retrain_df, prediction_df])

    final_df = create_retrain_table(retrain_df, dst_path)
    final_df.to_csv(join(dst_path, "retrain_results.csv"))
    return final_df


def collect_model_run_files(src_path: str, criteria_rank: str, files: List[str]) -> pd.DataFrame:
    config = yaml.load(open(join(src_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    dp_c = ["drop_columns", "drop_prefixes", "task", "search", "ground_truth", "label_mean", "label_std"]
    for k in dp_c:
        del config[k]

    result_files = []
    for model_file in list(filter(lambda x: "model__" in x, files)):
        model_df = pd.read_csv(join(src_path, model_file), index_col=0)
        best_combination = model_df.sort_values(by=criteria_rank, ascending=True).iloc[0]
        best_combination = best_combination[best_combination.index.str.contains("mean_test|std_test|rank_")]
        config["model"] = model_file.replace("model__", "").replace(".csv", "")
        config["result_path"] = src_path
        config["model_file"] = model_file
        result_files.append(pd.concat([best_combination, pd.Series(config)]))

    result_df = pd.DataFrame.from_records(result_files)
    return result_df


def retrain_model(result_path: str, model_file: str, dst_path: str) -> pd.DataFrame:
    model_name = model_file.replace("model__", "").replace(".csv", "")
    result_filename = join(dst_path, f"{model_name}_results.csv")
    if exists(result_filename):
        logging.info(f"Skip re-training of {model_name.upper()} as result already exists.")
        return pd.read_csv(result_filename, index_col=0)

    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    res_df = pd.read_csv(join(result_path, model_file))
    model = instantiate_best_model(res_df, model_name, config["task"])

    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)

    logging.info(f"Re-train best {model_name.upper()} model from path {result_path}.")
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


def collect_retrain_results(src_path: str, file_name: str) -> pd.DataFrame:
    file_locations = []
    for root, dirs, files in os.walk(src_path):
        for file in files:
            if file == file_name:
                result_df = pd.read_csv(join(root, file), index_col=0)
                # temp_context = basename(root).split("_")[-1]
                # result_df["temporal_context"] = int(temp_context) if temp_context != "False" else 0
                file_locations.append(result_df)

    return pd.concat(file_locations, axis=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, dest="data_path", default="data/training")
    parser.add_argument("--result_path", type=str, dest="result_path", default="results/ml/train")
    parser.add_argument("--exp_path", type=str, dest="exp_path", default="experiments/ml")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="results/ml/test")
    parser.add_argument("--exp_folder", type=str, dest="exp_folder", default="results/ml/train/2023-12-22-11-05-43")
    parser.add_argument("--train", type=bool, dest="train", default=False)
    parser.add_argument("--eval", type=bool, dest="eval", default=True)
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
        # evaluate_entire_training_folder(args.exp_folder, aggregate=True)

        # imu_df = pd.read_csv("results/ml/test/2023-11-16-16-45-14/rpe_imu/total_run_results.csv", index_col=0)
        # kinect_df = pd.read_csv("results/ml/test/2023-11-16-16-45-14/rpe_kinect/total_run_results.csv", index_col=0)
        # both_df = pd.read_csv("results/ml/test/2023-11-16-16-45-14/rpe_both/total_run_results.csv", index_col=0)
        #
        # result_df = pd.concat([imu_df, kinect_df, both_df], axis=1, keys=["IMU", "Kinect", "Both"],
        #                       names=["Metrics", None])
        # result_df = result_df.swaplevel(0, 1, axis=1)
        # result_df = result_df.sort_index(axis=1, level=0, ascending=False)
        # result_df.to_latex("results/ml/test/2023-11-16-16-45-14/total_run_results_latex.txt", escape=False)

        hrv_df = pd.read_csv("results/ml/test/2023-12-22-11-05-43/rpe_hrv/total_run_results.csv", index_col=0)
        flywheel_df = pd.read_csv("results/ml/test/2023-12-23-11-51-48/rpe_flywheel/total_run_results.csv", index_col=0)
        both_df = pd.read_csv("results/ml/test/2023-12-23-11-51-48/rpe_fusionbase/total_run_results.csv", index_col=0)

        result_df = pd.concat([hrv_df, flywheel_df, both_df], axis=1, keys=["HRV", "Flywheel", "Both"],
                              names=["Metrics", None])
        result_df = result_df.swaplevel(0, 1, axis=1)
        result_df = result_df.sort_index(axis=1, level=0, ascending=False)
        result_df.to_latex("total_run_results_latex.txt", escape=False, float_format="%.2f")
