import numpy as np
import pandas as pd
import tensorflow as tf
import math

from typing import Union, Tuple
from imblearn.over_sampling import RandomOverSampler


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
            meta_data: bool = False,
    ):
        assert len(X) == len(y), "X and y must have the same length"

        if label_col not in y.columns:
            raise ValueError(f"Label column {label_col} not in y.")

        if meta_data and not any([c in y.columns for c in ["set_id", "subject"]]):
            raise ValueError("Meta data is not in y.")

        self._X = X
        if meta_data:
            self._y = y[[label_col, "set_id", "subject"]].values
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
        print(self)

    def on_epoch_end(self):
        self._build_index()

    def _build_index(self):
        indices = []
        labels = []
        for file_counter, (input_arr, label) in enumerate(zip(self._X, self._y)):
            n_windows = math.floor((len(input_arr) - self._win_size) / self._stride) + 1
            indices.extend([(file_counter, i * self._stride) for i in range(n_windows)])
            labels.extend([label] * n_windows)
            # print(n_windows, len(indices))

        indices = np.array(indices)
        labels = np.array(labels).reshape(len(indices), -1)
        self._index = np.hstack((np.array(indices), np.array(labels)))

        if self._balance:
            indices, labels = self._index[:, :-1], self._index[:, -1]
            indices, labels = RandomOverSampler().fit_resample(indices, labels)
            self._index = np.hstack((indices, labels.reshape(-1, 1)))

        if self._shuffle:
            np.random.shuffle(self._index)

    def __getitem__(self, idx: int):
        X, y = [], []
        for ele in self._index[idx * self._batch_size:idx * self._batch_size + self._batch_size]:
            if len(ele) == 3:
                file_c, win_idx, label = int(ele[0]), int(ele[1]), ele[2]
            elif len(ele) > 4:
                file_c, win_idx, label = int(ele[0]), int(ele[1]), ele[2:]
            else:
                raise ValueError(f"Invalid index element: {ele}")

            X.append(self._X[file_c][win_idx:win_idx + self._win_size])
            y.append(label)

        X_batch = np.array(X)
        y_batch = np.array(y)
        return X_batch, y_batch

    def __len__(self):
        return self._n_samples // self._batch_size

    def __repr__(self):
        return f"SequenceGenerator: Got samples: {self._n_samples} in {len(self)} batches."


class DualSequenceGenerator(object):

    def __init__(
            self,
            X: Tuple[np.ndarray, np.ndarray],
            y: Union[np.ndarray, pd.DataFrame],
            sampling_frequencies: Tuple[int, int],
            label_col: str,
            win_size: int,  # in seconds
            overlap: float,
            batch_size: int,
            shuffle: bool = True,
            balance: bool = False,
            deliver_sets: bool = False,
    ):
        super().__init__(X[0], y, label_col, win_size, overlap, batch_size, shuffle, balance, deliver_sets)
        self._X2 = X[1]
        self._index_factor = 30 / 128

    def __getitem__(self, idx: int):
        X1, X2, y = [], [], []
        for file_c, win_idx, label in self._index[idx * self._batch_size:idx * self._batch_size + self._batch_size]:
            file_c = int(file_c)
            win_idx = int(win_idx)
            X1.append(self._X[file_c][win_idx:win_idx + self._win_size])
            y.append(label)

        X1_batch = np.array(X1)
        X2_batch = np.array(X2)
        y_batch = np.array(y)
        return X1_batch, X2_batch, y_batch


if __name__ == '__main__':
    X_imu = np.load("../../data/training/X_imu.npz", allow_pickle=True)["X"]
    X_kinect = np.load("../../data/training/X_kinect.npz", allow_pickle=True)["X"]
    y = pd.read_csv("../../data/training/y.csv", index_col=0)

    print(len(X_kinect))
    gen = SequenceGenerator(
        X_imu, y, label_col="rpe", win_size=384, overlap=0.9, batch_size=16, shuffle=True, balance=False,
        meta_data=False,
    )
    print(gen[0])

    # gen = SequenceGenerator(
    #     X_kinect, y, label_col="rpe", win_size=90, overlap=0.9, batch_size=16,
    #     shuffle=True, balance=False, deliver_sets=False,
    # )

    # for batch_idx in range(len(gen)):
    #     x, y = gen[batch_idx]
    #     if len(y.shape) == 2:
    #         print(y[:, 0])
    #     print(x.shape, y.shape)
