from .data_loader import DataLoader
from os.path import join

import os


class RawDataIterator(object):

    def __init__(self, input_path):
        self._input_path = input_path
        self._data_loaders = self.load_config_files()
        print(f"Found {len(self._data_loaders)} subject folders.")

    def load_config_files(self):
        """
        Function checks for valid configuration files
        @return: a list that contains all configuration files
        """
        data_loaders = []

        for subject in os.listdir(self._input_path):
            loader = DataLoader(join(self._input_path, subject))
            data_loaders.append(loader)

        return data_loaders

    def iterate_over_all_subjects(self):
        """
        Method returns iterator over sets in entire data set
        @return: iterator
        """
        for data_loader in self._data_loaders:
            for trial in data_loader.iterate_over_sets():
                yield trial
