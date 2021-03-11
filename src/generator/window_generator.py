from os.path import join

import os
import pandas as pd
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
        self.indices = {}
        self.data = {}
        for counter, cur_set in enumerate(sets):
            azure = pd.read_csv(join(base_path, cur_set, "azure.csv"), sep=";")
            imu = pd.read_csv(join(base_path, cur_set, "gaitup.csv"), sep=";")

            n_samples = int((len(azure) - 30 + n_steps) / n_steps)
            n_samples_o = int((len(imu) - 128 + 12.8) / 12.8)
            print(n_samples, n_samples_o)
            # print(len(azure) / len(imu))
            # print(len(azure), len(imu))
            self.data[counter] = (azure, imu)
            self.indices[counter] = (np.arange(0, len(azure)), np.arange(len(imu)))

    def generate_sliding_windows(self, n_epochs):
        for epoch in range(n_epochs):
            print(f"Epoch nr: {epoch} started.")
            yield epoch


if __name__ == '__main__':
    gen = Generator("../../data/intermediate", 3)
