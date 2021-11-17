from src.features import collect_all_trials_with_labels
from src.ml import split_data_based_on_pseudonyms

from argparse import ArgumentParser
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from os.path import join

import tensorflow as tf

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="results/2021-10-07-15-07-19")
parser.add_argument('--data_path', type=str, dest='data_path', default="data/processed")
parser.add_argument('--n_features', type=int, dest='n_features', default=51)
parser.add_argument('--n_frames', type=int, dest='n_frames', default=60)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
args = parser.parse_args()

# model = tf.keras.models.load_model(join(args.src_path, "models"))
# model.summary()

X, y = collect_all_trials_with_labels(args.data_path)
X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X, y, train_p=0.7, random_seed=42)
X_train, y_train, X_val, y_val = split_data_based_on_pseudonyms(X_train, y_train, train_p=0.7, random_seed=42)

train_gen = TimeseriesGenerator(
    X_train.to_numpy(),
    y_train['rpe'].to_numpy(),
    length=args.n_frames,
    batch_size=args.batch_size,
    shuffle=True,
)

for x, y in train_gen:
    print(x.shape, y.shape)
