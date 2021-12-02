from src.features import collect_all_trials_with_labels
from src.dl import TimeSeriesGenerator
from src.ml import split_data_based_on_pseudonyms_multiple_inputs
from src.utils import create_folder_if_not_already_exists
from argparse import ArgumentParser
from os.path import join, basename, normpath

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="models/2021-12-02-12-46-16")
parser.add_argument('--data_path', type=str, dest='data_path', default="data/processed")
parser.add_argument('--dst_path', type=str, dest='dst_path', default="results")
parser.add_argument('--n_features', type=int, dest='n_features', default=51)
parser.add_argument('--n_frames', type=int, dest='n_frames', default=60)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
args = parser.parse_args()

model = tf.keras.models.load_model(join(args.src_path, "models"))
model.summary()

result_path = join(args.dst_path, basename(normpath(args.src_path)))
create_folder_if_not_already_exists(result_path)

X, y = collect_all_trials_with_labels(args.data_path)
X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms_multiple_inputs(X, y, train_p=0.75, random_seed=69)
X_train, y_train, X_val, y_val = split_data_based_on_pseudonyms_multiple_inputs(X_train, y_train, train_p=0.75, random_seed=69)


test_gen = TimeSeriesGenerator(
    X_test,
    y_test,
    batch_size=args.batch_size,
    win_size=1,
    overlap=0.9,
    shuffle=False,
    balance=False,
)

ground_truth = []
predictions = []

for X, y in test_gen:
    y_pred = model.predict(X)
    ground_truth.append(y.reshape(-1))
    predictions.append(y_pred.reshape(-1))

predictions = np.concatenate(predictions, axis=0)
ground_truth = np.concatenate(ground_truth, axis=0)

plt.plot(predictions, label="Predictions")
plt.plot(ground_truth, label="Ground Truth")
plt.plot(predictions - ground_truth, label="Error")
plt.legend()
plt.show()
plt.savefig(join(result_path, "performance.png"))
