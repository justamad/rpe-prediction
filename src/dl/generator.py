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
            n_samples: int = 1,
            stride: int = 1,
            batch_size: int = 32,
            balance: bool = False,
            shuffle: bool = False,
    ):
        if len(X) != len(y):
            raise AttributeError(f"Data and targets do not have same length: {len(X)} vs {len(y)}.")

        self.__n_samples = n_samples
        self.__stride = stride
        self.__batch_size = batch_size

        indices = []
        for x, rpe in zip(X, y.iterrows()):
            indices.extend(self.__calculate_indices(x, rpe[1]))

        self.__indices = pd.DataFrame(data=indices, columns=['win_0', 'win_1', 'name', 'rpe'])
        self.check_distribution()
        self.check_distribution()

    def __calculate_indices(self, X: np.ndarray, y: pd.Series):
        n_windows_0 = self.calculation_nr_windows(len(X[0]), 30, 30 // 2)
        n_windows_1 = self.calculation_nr_windows(len(X[1]), 128, 128 // 2)
        n_windows = min(n_windows_0, n_windows_1)
        return [[idx * (30 // 2), idx * (128 // 2), y['name'], y['rpe']] for idx in range(n_windows)]

    @staticmethod
    def calculation_nr_windows(length: int, win_size: int, stride: int):
        return math.floor((length - win_size) / stride) + 1

    def check_distribution(self):
        rpe_dist = self.__indices['rpe'].value_counts()
        max_rpe = max(rpe_dist)
        for rpe, freq in rpe_dist.items():
            df_rpe = self.__indices[self.__indices['rpe'] == rpe]
            while freq < max_rpe:
                missing_frames = min(max_rpe - freq, len(df_rpe))
                index_mask = np.random.choice(len(df_rpe), missing_frames, replace=False)
                self.__indices = pd.concat([self.__indices, df_rpe.iloc[index_mask]], ignore_index=True)
                freq += missing_frames

    def __len__(self):
        return len(self.__indices)
