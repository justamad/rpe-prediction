from os.path import join

import os
import pandas as pd
import copy
import random
import numpy as np


class Generator(object):

    def __init__(self, base_path, n_steps):
        """
        Generator class for time series data
        @param base_path: path to processed data
        @param n_steps: step size for sliding window
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Given path {base_path} does not exist.")

        sets = os.listdir(base_path)
        self.indices = []
        self.data = {}
        self._window_size = 30
        self._n_steps = n_steps
        for counter, cur_set in enumerate(sets):
            azure = pd.read_csv(join(base_path, cur_set, "azure.csv"), sep=";").to_numpy()
            imu = pd.read_csv(join(base_path, cur_set, "gaitup.csv"), sep=";").to_numpy()
            # print(azure.shape)
            n_samples = min(int((len(azure) - 30 + n_steps) / n_steps), int((len(imu) - 128 + 12.8) / 12.8))
            self.data[counter] = (azure, imu)
            self.indices.extend([(counter, i) for i in range(0, n_steps * n_samples, n_steps)])

    def generate_sliding_windows(self, n_epochs):
        indices = copy.deepcopy(self.indices)
        for epoch in range(n_epochs):
            print(f"Epoch nr: {epoch} started.")
            while len(indices) > 0:
                random_nr = random.randint(0, len(indices) - 1)
                set_nr, start_idx = indices.pop(random_nr)
                azure_data, imu_data = self.data[set_nr]
                batch = azure_data[start_idx:start_idx+self._window_size, :]
                yield batch


if __name__ == '__main__':
    gen = Generator("../../data/intermediate", 3)
    for w in gen.generate_sliding_windows(1):
        print(w.shape)
