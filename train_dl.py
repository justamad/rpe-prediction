from src.dl import build_fcn_regression_model, FixedLengthIterator, build_conv_lstm_regression_model
from src.dataset import get_subject_names_random_split, normalize_data_by_subject
from src.dataset import extract_dataset_input_output
from src.ml import MLOptimization, ConvLSTMModelConfig
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
        normalization_ground_truth: str,
        task: str,
):
    X, y = extract_dataset_input_output(df=df, ground_truth_column=ground_truth)
    X = X.values.reshape((-1, seq_length, 55))
    # y = y.loc[:, ground_truth].values.flatten()[::seq_length]
    y = y.iloc[::seq_length, :]

    # subjects = y["subject"].unique()
    # y["group"] = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))

    opti = MLOptimization(X=X, y=y, balance=False, task=task, mode=search, ground_truth=ground_truth)
    models = [ConvLSTMModelConfig()]
    opti.perform_grid_search_with_cv(models, log_path=log_path, n_jobs=1)

    # train_gen = FixedLengthIterator(X, y, ground_truth=ground_truth, fixed_length=seq_length, batch_size=batch_size)
    # print(train_gen)

    # grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=10)
    # grid_result = grid.fit(X, y)
    # grid_result = grid.fit(train_gen)
    # print(grid_result)

    # X_train, y_train = prepare_dataset(train_df)
    # X_test, y_test = prepare_dataset(test_df)

    # test_gen = FixedLengthIterator(X_test, y_test, batch_size=1)

    # print(train_gen, test_gen)

    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # model.fit(
    #     train_gen,
    #     validation_data=train_gen,
    #     epochs=epochs,
    #     callbacks=[tensorboard_callback],
    # )
    # model.save(model_name)
    return model


# def evaluate_model(model, test_df: pd.DataFrame, win_size: int, overlap: float):
#     X_test, y_test = prepare_dataset(test_df)
#     test_gen = DataSetIterator(X_test, y_test, win_size=win_size, overlap=overlap, batch_size=1, shuffle=False)
#     print(test_gen)
#     # model.evaluate(test_gen)
#
#     values = model.predict(test_gen)
#     plt.plot(values.reshape(-1))
#     plt.plot([test_gen[i][1].reshape(-1) for i in range(len(test_gen))])
#     plt.show()


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

    # # df = normalize_data_by_subject(df)
    # # df = normalize_subject_rpe(df)
    # train_mask = get_subject_names_random_split(df, train_p=0.7, random_seed=42)
    # train_df = df.loc[train_mask].copy()
    # test_df = df.loc[~train_mask].copy()
    #
    # print("Train Subjects:", train_df["subject"].unique())
    # print("Test Subjects:", test_df["subject"].unique())

    # if args.train:
    #     model = train_model(
    #         train_df=train_df,
    #         test_df=test_df,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         model_name=model_name,
    #     )
    # else:
    #     model = tf.keras.models.load_model(model_name)
    #
    # evaluate_model(model, test_df, win_size=args.win_size, overlap=args.overlap)
