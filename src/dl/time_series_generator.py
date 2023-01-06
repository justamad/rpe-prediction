from typing import List

import numpy as np
import pandas as pd
import math
import tensorflow as tf


class DataSetIterator(tf.keras.utils.Sequence):

    def __init__(
            self,
            X: List[pd.DataFrame],
            y: List[pd.DataFrame],
            label_column: str = None,
            batch_size: int = 32,
            win_size: float = 1.0,
            overlap: float = 0.5,
            balance: bool = True,
            shuffle: bool = True,
    ):
        if len(X) != len(y):
            raise AttributeError(f"Input and label dimensions do not match: {len(X)} vs {len(y)}.")

        self._X = X
        self._y = y
        self._stride = int(win_size - (win_size * overlap))
        self._win_size = win_size

        self._batch_size = batch_size
        self._balance = balance
        self._shuffle = shuffle
        self.__indices = []
        self.on_epoch_end()

    def __build_index(self):
        indices = []
        for data_idx, data_entry in enumerate(self._X):
            n_windows = math.floor((len(data_entry) - self._win_size) / self._stride) + 1
            indices.extend([(data_idx, win_idx * self._stride) for win_idx in range(n_windows)])

        return indices

    def __getitem__(self, index):
        X, y = [], []
        for n_sample in range(index, index + self._batch_size):
            data_idx, win_idx = self.__indices[n_sample]
            X.append(self._X[data_idx][win_idx:win_idx + self._win_size])
            y.append(self._y[data_idx][0][0])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X, y

    def on_epoch_end(self):
        self.__indices = self.__build_index()

    def __len__(self):
        return len(self.__indices) // self._batch_size

    def __repr__(self):
        return f"""
Generator TimeSeries: win_size: {self._win_size}, stride: {self._stride}, batch_size: {self._batch_size}
Number of batches: [{len(self)}]
"""


if __name__ == '__main__':
    df = pd.read_csv("../../data/processed/train_dl_test.csv", index_col=0)
    subsets = [df[df["set_id"] == i] for i in df["set_id"].unique()]
    subsets = [(df.iloc[:, :-3], df.iloc[:, -3:]) for df in subsets]
    X = list(map(lambda x: x[0].values, subsets))
    y = list(map(lambda x: x[1].values, subsets))

    generator = DataSetIterator(X, y, win_size=30, overlap=0.0, batch_size=8)
    print(generator)

    for i in range(len(generator)):
        print(f"iteration: {i}", generator[i][0].shape, generator[i][1].shape)
        # print(generator[i])
