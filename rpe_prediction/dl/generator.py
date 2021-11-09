import numpy as np
import math


class TimeSeriesGenerator(object):

    def __init__(
            self,
            X: list,
            y: list,
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
        self._indices = []

        for counter, (x, rpe) in enumerate(zip(X, y)):
            self._indices.extend(self.__calculate_indices(x, rpe, counter))

    def __calculate_indices(self, X: np.ndarray, y: int, counter: int):
        n_windows = math.floor((len(X) - self.__n_samples) / self.__stride) + 1
        return [[counter, idx * self.__stride, y] for idx in range(n_windows)]

    def __len__(self):
        pass
