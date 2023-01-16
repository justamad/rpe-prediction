from src.dl import build_fcn_regression_model, DataSetIterator
from src.dataset import get_subject_names_random_split, normalize_data_by_subject
from argparse import ArgumentParser

import pandas as pd
import tensorflow as tf
import datetime
import os


def prepare_dataset(df: pd.DataFrame):
    subsets = [df[df["set_id"] == i] for i in df["set_id"].unique()]
    subsets = [(df.iloc[:, :-3], df.iloc[:, -3:]) for df in subsets]
    X = list(map(lambda x: x[0].values, subsets))
    y = list(map(lambda x: x[1].values, subsets))
    return X, y


def train_model(train_df, test_df, epochs: int, batch_size: int, win_size: int, overlap: float):
    X_train, y_train = prepare_dataset(train_df)
    X_test, y_test = prepare_dataset(test_df)

    train_gen = DataSetIterator(X_train, y_train, win_size=win_size, overlap=overlap, batch_size=batch_size)
    test_gen = DataSetIterator(X_test, y_test, win_size=win_size, overlap=overlap, batch_size=batch_size)

    print(train_gen, test_gen)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model = build_fcn_regression_model(n_samples=win_size, n_features=33)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse"])
    model.summary()
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        callbacks=[tensorboard_callback],
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_file", type=str, dest="src_file", default="data/processed/dl_ori.csv")
    parser.add_argument("--win_size", type=int, dest="win_size", default=90)
    parser.add_argument("--batch_size", type=int, dest="batch_size", default=16)
    parser.add_argument("--epochs", type=int, dest="epochs", default=10)
    parser.add_argument("--overlap", type=float, dest="overlap", default=0.95)
    parser.add_argument("--log_path", type=str, dest="log_path", default="results")
    parser.add_argument("--dst_path", type=str, dest="dst_path", default="data/processed")
    parser.add_argument("--show", type=bool, dest="show", default=True)
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    df = pd.read_csv(args.src_file, index_col=0)
    df = normalize_data_by_subject(df)
    train_mask = get_subject_names_random_split(df, train_p=0.8, random_seed=42)
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()

    print("Train Subjects:", train_df["subject"].unique())
    print("Test Subjects:", test_df["subject"].unique())

    train_model(
        train_df=train_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        win_size=args.win_size,
        overlap=args.overlap,
    )
