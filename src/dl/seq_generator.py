import numpy as np
import pandas as pd
import tensorflow as tf
import math

from imblearn.over_sampling import RandomOverSampler
from typing import Union


class SequenceGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            X: np.ndarray,
            y: Union[np.ndarray, pd.DataFrame],
            label_col: str,
            win_size: int,
            overlap: float,
            batch_size: int,
            shuffle: bool = True,
            balance: bool = False,
            deliver_sets: bool = False,
    ):
        assert len(X) == len(y), "X and y must have the same length"

        if label_col not in y.columns:
            raise ValueError(f"Label column {label_col} not in y")

        if deliver_sets and "set_id" not in y.columns:
            raise ValueError("If deliver_sets is True, y must have a column named set")

        self._X = X
        if deliver_sets:
            self._y = y[[label_col, "set_id"]].values
        else:
            self._y = y[label_col].values

        self._win_size = win_size
        self._stride = int(self._win_size - (self._win_size * overlap))
        self._shuffle = shuffle
        self._balance = balance
        self._index = []

        self._label_col = label_col
        self._batch_size = batch_size

        self._build_index()
        self._n_samples = len(self._index)

        print("Got samples: ", self._n_samples)
        print(f"Number of batches: {len(self)}")

    def on_epoch_end(self):
        self._build_index()

    def _build_index(self):
        indices = []
        for file_counter, (input_arr, label) in enumerate(zip(self._X, self._y)):
            n_windows = math.floor((len(input_arr) - self._win_size) / self._stride) + 1
            indices.extend([(file_counter, i, label) for i in range(0, n_windows, self._stride)])

        self._index = np.array(indices)

        if self._balance:
            indices, labels = self._index[:, :-1], self._index[:, -1]
            indices, labels = RandomOverSampler().fit_resample(indices, labels)
            self._index = np.hstack((indices, labels.reshape(-1, 1)))

        if self._shuffle:
            np.random.shuffle(self._index)

    def __getitem__(self, idx: int):
        X, y = [], []
        for file_c, win_idx, label in self._index[idx * self._batch_size:idx * self._batch_size + self._batch_size]:
            file_c = int(file_c)
            win_idx = int(win_idx)
            X.append(self._X[file_c][win_idx:win_idx + self._win_size])
            y.append(label)

        X_batch = np.array(X)
        y_batch = np.array(y)
        return X_batch, y_batch

    def __len__(self):
        return self._n_samples // self._batch_size


if __name__ == '__main__':
    X = np.load("../../data/training/X_lstm.npz", allow_pickle=True)["X"]
    y = pd.read_csv("../../data/training/y_lstm.csv", index_col=0)

    gen = SequenceGenerator(
        X, y, label_col="Mean HR (1/min)", win_size=150, overlap=0.5, batch_size=4, shuffle=True, balance=False,
        deliver_sets=False
    )

    for batch_idx in range(len(gen)):
        x, y = gen[batch_idx]
        if len(y.shape) == 2:
            print(y[:, 0])
        print(x.shape, y.shape)
