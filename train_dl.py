from src.dl import regression_models
from src.ml import MLOptimization
from src.dataset import get_subject_names_random_split, normalize_data_by_subject
from src.dataset import extract_dataset_input_output
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor
from argparse import ArgumentParser
from os.path import join

import pandas as pd
import tensorflow as tf
import yaml
import datetime
import os
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def train_model(
        df: pd.DataFrame,
        log_path: str,
        ground_truth: str,
        seq_length: int,
        normalization: str,
        search: str,
        balancing: bool,
        normalization_labels: str,
        task: str,
):
    X, y = extract_dataset_input_output(df=df, ground_truth_column=ground_truth)
    X = X.values.reshape((-1, seq_length, 55))
    # y = y.loc[:, ground_truth].values.flatten()[::seq_length]
    y = y.iloc[::seq_length, :]

    ml_optimization = MLOptimization(X=X, y=y, balance=False, task=task, mode=search, ground_truth=ground_truth)
    ml_optimization.perform_grid_search_with_cv(regression_models, log_path=log_path, n_jobs=1)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model.fit(
    #     train_gen,
    #     validation_data=train_gen,
    #     epochs=epochs,
    #     callbacks=[tensorboard_callback],
    # )
    # model.save(model_name)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="data/training")
    parser.add_argument("--log_path", type=str, dest="log_path", default="results")
    parser.add_argument("--run_experiments", type=str, dest="run_experiments", default="experiments_dl")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.run_experiments:
        for ground_truth in os.listdir(args.run_experiments):
            exp_path = join(args.run_experiments, ground_truth)
            for config_file in filter(lambda f: not f.startswith("_"), os.listdir(exp_path)):
                exp_config = yaml.load(open(join(exp_path, config_file), "r"), Loader=yaml.FullLoader)
                df = pd.read_csv(join(args.src_path, exp_config["training_file"]), index_col=0)
                del exp_config["training_file"]
                train_model(df, args.log_path, **exp_config)
                # evaluate_for_specific_ml_model(eval_path)
