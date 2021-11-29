from typing import List

import numpy as np
import pandas as pd
import math


class TimeSeriesGenerator(object):

    def __init__(
            self,
            X: List,
            y: pd.DataFrame,
            sampling_frequencies: List,
            n_epochs: int = 1,
            overlap: float = 0.5,
            batch_size: int = 32,
    ):
        if len(X) != len(y):
            raise AttributeError(f"Data and targets do not have same length: {len(X)} vs {len(y)}.")

        self.__n_epochs = n_epochs
        self.__overlap = overlap
        self.__batch_size = batch_size
        self._X = X
        self._y = y

        self._kinect_stride = int(30 * self.__overlap)
        self._imu_stride = int(128 * self.__overlap)

        indices = []
        for win_id, (x, rpe) in enumerate(zip(X, y.iterrows())):
            indices.extend(self.__calculate_indices(x, rpe[1], win_id))

        self.__indices = pd.DataFrame(data=indices, columns=['win_id', 'win_0', 'win_1', 'name', 'rpe'])

    def get_iterator(self):
        for cur_epoch in range(self.__n_epochs):
            idx_df = self._balance_sample_distribution()
            indices = np.array(idx_df.index)
            np.random.shuffle(indices)

            while len(indices) > 0:
                X_data = []
                y_data = []

                cut_idx = indices[0:self.__batch_size]
                cur_batch = idx_df.iloc[cut_idx]
                for _, (win_id, win_0, win_1, name, rpe) in cur_batch.iterrows():
                    data_frames = self._X[win_id]
                    kinect = data_frames[0]
                    imu = data_frames[1]

                    X_data.append((
                        kinect.iloc[win_0:win_0 + self._kinect_stride],
                        imu.iloc[win_1:win_1 + self._imu_stride],
                    ))

                    y_data.append(rpe)

                yield X_data, np.array(y_data)
                indices = indices[self.__batch_size:]

    def __calculate_indices(self, X: np.ndarray, y: pd.Series, win_id: int):
        n_windows_0 = self.calculation_nr_windows(len(X[0]), 30, self._kinect_stride)
        n_windows_1 = self.calculation_nr_windows(len(X[1]), 128, self._imu_stride)
        n_windows = min(n_windows_0, n_windows_1)
        return [[win_id, idx * (30 // 2), idx * (128 // 2), y['name'], y['rpe']] for idx in range(n_windows)]

    @staticmethod
    def calculation_nr_windows(length: int, win_size: int, stride: int):
        return math.floor((length - win_size) / stride) + 1

    def _balance_sample_distribution(self) -> pd.DataFrame:
        indices = self.__indices.copy()
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
        return len(self.__indices)
