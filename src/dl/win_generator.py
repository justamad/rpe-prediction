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
        # self.n_name = df[y_col['name']].nunique()
        # self.n_type = df[y_col['type']].nunique()

    def _build_index(self):
        self._index.clear()
        for file_counter, (input, label) in enumerate(zip(self._X, self._y)):
            length = len(input)
            n_windows = math.floor((length - self._win_size) / self._stride) + 1
            self._index.extend([(file_counter, i, label) for i in range(n_windows)])

        if self._shuffle:
            np.random.shuffle(self._index)

        if self._balance:
            raise NotImplementedError("Balancing is not implemented yet")

    def on_epoch_end(self):
        self._build_index()

    def __getitem__(self, index: int):
        X, y = [], []
        for _ in range(self._batch_size):
            file_counter, window_idx, label = self._index.pop(0)
            X.append(self._X[file_counter][window_idx:window_idx + self._win_size])
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
