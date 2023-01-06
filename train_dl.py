from src.dl import build_fcn_regression_model, DataSetIterator
from src.dataset import get_subject_names_random_split
from argparse import ArgumentParser

import pandas as pd
import tensorflow as tf
import datetime
import os


parser = ArgumentParser()
parser.add_argument("--win_size", type=int, dest="win_size", default=60)
parser.add_argument("--batch_size", type=int, dest="batch_size", default=32)
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

df = pd.read_csv("data/processed/train_dl_test.csv", index_col=0)
train_mask = get_subject_names_random_split(df, train_p=0.8, random_seed=42)
df_train = df.loc[train_mask].copy()
df_test = df.loc[~train_mask].copy()


def prepare_dataset(df: pd.DataFrame):
    subsets = [df[df["set_id"] == i] for i in df["set_id"].unique()]
    subsets = [(df.iloc[:, :-3], df.iloc[:, -3:]) for df in subsets]
    X = list(map(lambda x: x[0].values, subsets))
    y = list(map(lambda x: x[1].values, subsets))
    return X, y


X_train, y_train = prepare_dataset(df_train)
X_test, y_test = prepare_dataset(df_test)


train_gen = DataSetIterator(X_train, y_train, win_size=args.win_size, overlap=0.9, batch_size=args.batch_size)
test_gen = DataSetIterator(X_test, y_test, win_size=args.win_size, overlap=0.9, batch_size=args.batch_size)

print(train_gen, test_gen)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model = build_fcn_regression_model(n_samples=60, n_features=96)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse"])
model.summary()
model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=10,
    callbacks=[tensorboard_callback]
)
