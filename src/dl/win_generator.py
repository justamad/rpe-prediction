import numpy as np
import pandas as pd
import tensorflow as tf
import math


class WinDataGen(tf.keras.utils.Sequence):

    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            win_size: int,
            overlap: float,
            batch_size: int,
            shuffle: bool = True,
            balance: bool = False
    ):
        assert len(X) == len(y), "X and y must have the same length"
        self._X = X
        self._y = y
        self._win_size = win_size
        self._stride = int(self._win_size - (self._win_size * overlap))
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._balance = balance
        self._index = []
        self._build_index()
        self._n_samples = len(self._index)
        print("Got samples: ", self._n_samples)
        print(f"Number of batches: {len(self)}")
        # self.n_name = df[y_col['name']].nunique()
        # self.n_type = df[y_col['type']].nunique()

    def on_epoch_end(self):
        self._build_index()

    def _build_index(self):
        indices = []
        for file_counter, (input_arr, label) in enumerate(zip(self._X, self._y)):
            n_windows = math.floor((len(input_arr) - self._win_size) / self._stride) + 1
            indices.extend([(file_counter, i, label) for i in range(0, n_windows, self._stride)])

        self._index = np.array(indices)

        if self._shuffle:
            np.random.shuffle(self._index)

        if self._balance:
            raise NotImplementedError("Balancing is not implemented yet")

    def __getitem__(self, idx: int):
        X, y = [], []
        for file_c, win_idx, label in self._index[idx * self._batch_size:idx * self._batch_size + self._batch_size]:
            X.append(self._X[file_c][win_idx:win_idx + self._win_size])
            y.append(label)

        return np.array(X), np.array(y)

    def __len__(self):
        return self._n_samples // self._batch_size


if __name__ == '__main__':
    X = np.load("../../data/training/X_lstm.npz", allow_pickle=True)["X"]
    y = pd.read_csv("../../data/training/y_lstm.csv", index_col=0)
    y = y["rpe"]
    gen = WinDataGen(X, y, win_size=30, overlap=0.5, batch_size=4, shuffle=True, balance=False)
    for batch_idx in range(len(gen)):
        x, y = gen[batch_idx]
        print(x.shape, y.shape)
