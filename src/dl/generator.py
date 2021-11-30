from typing import List
from tensorflow.keras.utils import Sequence

import numpy as np
import pandas as pd
import math
import logging


def calculation_nr_windows(length: int, win_size: int, stride: int):
    return math.floor((length - win_size) / stride) + 1


class TimeSeriesGenerator(Sequence):

    def __init__(
            self,
            X: List,
            y: pd.DataFrame,
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
        self._overlap = overlap
        self._batch_size = batch_size
        self._balance = balance
        self._shuffle = shuffle

        self._kinect_size = int(win_size * 30)
        self._imu_size = int(win_size * 128)

        self._kinect_stride = int(self._kinect_size * (1 - self._overlap))
        self._imu_stride = int(self._imu_size * (1 - self._overlap))

        indices = []
        for win_id, (x, rpe) in enumerate(zip(X, y.iterrows())):
            indices.extend(self.__calculate_indices(x, rpe[1], win_id))

        self.__orig_indices = pd.DataFrame(data=indices, columns=['win_id', 'win_0', 'win_1', 'name', 'rpe'])

        if self._balance:
            self._idx_df = self._balance_sample_distribution()
        else:
            self._idx_df = self.__orig_indices

        self._cur_indices = np.array(self._idx_df.index)

        if self._shuffle:
            np.random.shuffle(self._cur_indices)

        logging.info(f"TimeSeriesGenerator: total samples={len(self)} "
                     f"win X1={self._kinect_size}, stride X1={self._kinect_stride} " 
                     f"win X2={self._imu_size}, stride X2={self._imu_stride}")

    def __getitem__(self, index):
        if index < 0:
            raise AttributeError("Index negative")

        X1_data = []
        X2_data = []
        y_data = []

        cur_index = self._cur_indices[index * self._batch_size: index * self._batch_size + self._batch_size]
        cur_batch = self._idx_df.iloc[cur_index]
        for _, (win_id, win_0, win_1, name, rpe) in cur_batch.iterrows():
            data_frames = self._X[win_id]
            kinect = data_frames[0]
            imu = data_frames[1]

            X1_data.append(kinect.iloc[win_0:win_0 + self._kinect_size])
            X2_data.append(imu.iloc[win_1:win_1 + self._imu_size])
            y_data.append(rpe)

        X1_data = np.array(X1_data)  # To clear numpy warnings...
        X2_data = np.array(X2_data)
        y = np.array(y_data).reshape(-1, 1)
        return [X1_data, X2_data], y

    def __calculate_indices(self, X: np.ndarray, y: pd.Series, win_id: int):
        n_windows = min(
            calculation_nr_windows(len(X[0]), self._kinect_size, self._kinect_stride),
            calculation_nr_windows(len(X[1]), self._imu_size, self._imu_stride),
        )
        return [[win_id, idx * self._kinect_stride, idx * self._imu_stride, y['name'], y['rpe']] for idx in range(n_windows)]

    def on_epoch_end(self):
        if self._balance:
            self._idx_df = self._balance_sample_distribution()
            self._cur_indices = np.array(self._idx_df.index)

        if self._shuffle:
            np.random.shuffle(self._cur_indices)

    def _balance_sample_distribution(self) -> pd.DataFrame:
        indices = self.__orig_indices.copy()
        rpe_dist = indices['rpe'].value_counts()
        max_rpe = max(rpe_dist)
        for rpe, freq in rpe_dist.items():
            df_rpe = indices[indices['rpe'] == rpe]
            while freq < max_rpe:
                missing_frames = min(max_rpe - freq, len(df_rpe))
                index_mask = np.random.choice(len(df_rpe), missing_frames, replace=False)
                indices = pd.concat([indices, df_rpe.iloc[index_mask]], ignore_index=True)
                freq += missing_frames

        return indices

    def __len__(self):
        return math.ceil(len(self._cur_indices) / self._batch_size)

    @property
    def get_x1_dim(self):
        return self._kinect_size

    @property
    def get_x2_dim(self):
        return self._imu_size
