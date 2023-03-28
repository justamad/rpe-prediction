from src.ml import MLOptimization
from src.dl import build_regression_models, build_conv_lstm_regression_model, instantiate_best_dl_model
from src.plot import evaluate_aggregated_predictions, evaluate_sample_predictions
from typing import List
from src.dataset import extract_dataset_input_output, normalize_data_by_subject, normalize_data_global
from datetime import datetime
from argparse import ArgumentParser
from sklearn.metrics import r2_score
from os.path import join, exists
from os import makedirs

import pandas as pd
import numpy as np
import logging
import tensorflow as tf
import yaml
import os
# import matplotlib
# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def train_model(
        df: pd.DataFrame,
        log_path: str,
        seq_length: int,
        drop_columns: List,
        drop_prefixes: List,
        ground_truth: str,
        normalization_input: str,
        search: str,
        balancing: bool,
        normalization_labels: str,
        task: str,
):
    if drop_prefixes is None:
        drop_prefixes = []
    if drop_columns is None:
        drop_columns = []

    X, y = extract_dataset_input_output(df=df, ground_truth_column=ground_truth)
    for prefix in drop_prefixes:
        drop_columns += [col for col in df.columns if col.startswith(prefix)]

    X.drop(columns=drop_columns, inplace=True, errors="ignore")
    X = X.loc[:, (X != 0).any(axis=0)]  # Remove columns with all zeros, e.g. constrained joint orientations

    # For quick checking...
    # subjects = y["subject"].unique()
    # mask = y["subject"].isin(subjects[:3])
    # X = X.loc[mask]
    # y = y.loc[mask]

    # Normalization
    input_mean, input_std = float('inf'), float('inf')
    label_mean, label_std = float('inf'), float('inf')

    if normalization_input == "subject":
        X = normalize_data_by_subject(X, y)
    elif normalization_input == "global":
        X = normalize_data_global(X)
    else:
        raise AttributeError(f"Unknown normalization method: {normalization_input}")

    X.to_csv(join(log_path, "X.csv"))
    y.to_csv(join(log_path, "y.csv"))

    X = X.values.reshape((-1, seq_length, X.shape[1]))
    y = y.iloc[::seq_length, :]

    # oversample = SMOTE()
    # X, y = oversample.fit_resample(X, y["rpe"])

    with open(join(log_path, "config.yml"), "w") as f:
        yaml.dump(
            {
                "task": task,
                "search": search,
                "seq_length": seq_length,
                "drop_columns": drop_columns,
                "ground_truth": ground_truth,
                "drop_prefixes": drop_prefixes,
                "normalization_input": normalization_input,
                "balancing": balancing,
                "normalization_labels": normalization_labels,
                "input_mean": float(input_mean),
                "input_std": float(input_std),
                "label_mean": float(label_mean),
                "label_std": float(label_std),
            },
            f,
        )

    # model = build_conv_lstm_regression_model(X.shape[1], X.shape[2])
    # model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["mse", "mae"])
    # model.summary()
    # model.fit(X, y["rpe"], epochs=10, batch_size=16)
    # y = model.predict(X)
    # plt.plot(y)
    # plt.show()
    # print(y)

    models = build_regression_models(X.shape[1], X.shape[2])
    ml_optimization = MLOptimization(X=X, y=y, balance=False, task=task, mode=search, ground_truth=ground_truth)
    ml_optimization.perform_grid_search_with_cv(models, log_path=log_path, n_jobs=1)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model.fit(
    #     train_gen,
    #     validation_data=train_gen,
    #     epochs=epochs,
    #     callbacks=[tensorboard_callback],
    # )
    # model.save(model_name)


def evaluate_ml_model(result_path: str):
    config = yaml.load(open(join(result_path, "config.yml"), "r"), Loader=yaml.FullLoader)
    X = pd.read_csv(join(result_path, "X.csv"), index_col=0)
    y = pd.read_csv(join(result_path, "y.csv"), index_col=0)

    if config["task"] == "classification":
        score_metric = "mean_test_f1_score"
    else:
        score_metric = "mean_test_r2"

    X = X.values.reshape((-1, config["seq_length"], X.shape[1]))
    y = y.iloc[::config["seq_length"], :]

    for model_file in list(filter(lambda x: x.startswith("model__"), os.listdir(result_path))):
        model_name = model_file.replace("model__", "").replace(".csv", "")
        logging.info(f"Evaluating model: {model_name}")
        result_df = pd.read_csv(join(result_path, model_file))

        result_dicts = {}
        r2_scores = []
        for subject in y["subject"].unique():
            model, meta_params = instantiate_best_dl_model(
                result_df, model_name=model_name, metric=score_metric, n_samples=X.shape[1], n_features=X.shape[2]
            )

            subject_mask = y["subject"] == subject
            X_train, y_train = X[~subject_mask], y[~subject_mask]
            X_test, y_test = X[subject_mask], y[subject_mask]

            model.fit(X_train, y_train["rpe"], **meta_params)
            y_pred = model.predict(X_test)
            y_test["predictions"] = y_pred
            r2_scores.append(r2_score(y_test["rpe"], y_test["predictions"]))
            result_dicts[subject] = y_test

        print(f"R2 score: {np.mean(r2_scores)}, {np.std(r2_scores)}")

        # evaluate_aggregated_predictions(result_dicts, config["ground_truth"], file_name=join(result_path, f"{model_name}_aggregated.png"))
        evaluate_sample_predictions(result_dicts, config["ground_truth"],
                                    file_name=join(result_path, f"{model_name}_sample.png"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results")
    parser.add_argument("--run_experiments", type=str, dest="run_experiments", default="experiments_dl")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # evaluate_ml_model("results/2023-03-27-17-32-05")

    if args.run_experiments:
        for exp_name in os.listdir(args.run_experiments):
            exp_path = join(args.run_experiments, exp_name)
            for config_file in filter(lambda f: not f.startswith("_"), os.listdir(exp_path)):
                exp_config = yaml.load(open(join(exp_path, config_file), "r"), Loader=yaml.FullLoader)
                training_file = exp_config["training_file"]
                df = pd.read_csv(join(args.src_path, training_file), index_col=0)
                del exp_config["training_file"]
                seq_len = int(training_file.split("_")[0])
                log_path = join(args.log_path, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

                if not exists(log_path):
                    makedirs(log_path)

                eval_path = train_model(df, log_path, seq_len, **exp_config)
                # evaluate_for_specific_ml_model(eval_path)
