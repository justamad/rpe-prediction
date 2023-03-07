from typing import List, Tuple
from keras import utils

import numpy as np
import pandas as pd


class FixedLengthIterator(utils.Sequence):

    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.DataFrame,
            fixed_length: int = 100,
            ground_truth: str = None,
            batch_size: int = 32,
            shuffle: bool = True,
            random_state: int = 42,
    ):
        if len(X) != len(y):
            raise AttributeError(f"Input and label dimensions do not match: {len(X)} vs {len(y)}.")

        if len(X) % fixed_length != 0:
            raise AttributeError(f"Input can not be divided by Fixed length argument: {len(X)} % {fixed_length} != 0")

        self._X = [X.iloc[i:i + fixed_length] for i in range(0, len(X), fixed_length)]
        self._y = [y.iloc[i:i + fixed_length] for i in range(0, len(y), fixed_length)]
        self._y = [df[ground_truth].unique()[0] for df in self._y]

        self._ground_truth = ground_truth
        self._batch_size = batch_size
        self._shuffle = shuffle
        if self._shuffle:
            np.random.seed(random_state)

        self.__indices = []
        self.on_epoch_end()

    def __build_index(self) -> List:
        indices = list(range(len(self._X)))
        if self._shuffle:
            np.random.shuffle(indices)

        return indices

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for n_sample in range(index, index + self._batch_size):
            idx = self.__indices[n_sample]
            X.append(self._X[idx])
            y.append(self._y[idx])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return X, y

    def on_epoch_end(self):
        self.__indices = self.__build_index()

    def __len__(self):
        return len(self.__indices) // self._batch_size

    def __repr__(self):
        return f"Iterator with {len(self._data) / self._batch_size} batches."


if __name__ == '__main__':
    test_df = pd.read_csv("../../data/training/padding.csv", index_col=0)

    generator = FixedLengthIterator(test_df, None, fixed_length=140, batch_size=16)
    print(generator)

    for i in range(len(generator)):
        X, y = generator[i]
        print(f"iteration: {i}", X.shape, y.shape)
