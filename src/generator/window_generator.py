from os.path import join

import os
import pandas as pd
import copy
import random
import numpy as np


class Generator(object):

    def __init__(self, base_path, n_steps, window_size, batch_size):
        """
        Generator class for time series data
        @param base_path: path to processed data
        @param n_steps: step size for sliding window
        @param window_size: current number of samples per window
        @param batch_size: the given batch size
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Given path {base_path} does not exist.")

        sets = os.listdir(base_path)
        self.indices = []
        self.data = {}
        self._window_size = window_size
        self._n_steps = n_steps
        self._batch_size = batch_size
        for counter, cur_set in enumerate(sets):
            azure = pd.read_csv(join(base_path, cur_set, "azure.csv"), sep=";").to_numpy()
            imu = pd.read_csv(join(base_path, cur_set, "gaitup.csv"), sep=";").to_numpy()
            n_samples = min(int((len(azure) - window_size + n_steps) / n_steps), int((len(imu) - 128 + 12.8) / 12.8))
            with open(join(base_path, cur_set, "label.txt")) as file:
                label = file.readline()
                label = float(label)

            self.data[counter] = (azure, imu, label)
            self.indices.extend([(counter, i) for i in range(0, n_steps * n_samples, n_steps)])

    def generate_sliding_windows(self, n_epochs):
        for epoch in range(n_epochs):
            print(f"Epoch nr: {epoch + 1} started.")

            indices = copy.deepcopy(self.indices)  # Better scramble data here
            random.shuffle(indices)

            current_batch, current_labels = [], []
            while len(indices) > 0:
                if len(current_batch) >= self._batch_size:
                    yield np.array(current_batch), np.array(current_labels)
                    current_batch, current_labels = [], []

                set_nr, start_idx = indices.pop()
                azure_data, imu_data, label = self.data[set_nr]
                data = azure_data[start_idx:start_idx+self._window_size, :]
                current_batch.append(data)
                current_labels.append(label)

            if len(current_batch) > 0:
                yield np.array(current_batch), np.array(current_labels)


if __name__ == '__main__':
    gen = Generator("../../data/intermediate", n_steps=3, window_size=30, batch_size=16)
    for samples, output in gen.generate_sliding_windows(2):
        print(samples.shape, output)
