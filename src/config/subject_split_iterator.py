from os.path import join
from .config_reader import ConfigReader

import os


class SubjectSplitIterator(object):

    def __init__(self, input_path, train_percentage: float = 0.7):
        self._input_path = input_path
        self._folders = os.listdir(self._input_path)
        print(f"Found the following folders: {self._folders}")
        # Perform split
        self._test_files = ["arne_flywheel"]
        self._train_files = ["justin"]

    def iterate_over_files(self, folders):
        for subject in folders:
            print(f"Deliver subject: {subject}")
            config_file = join(self._input_path, subject, "config.json")
            if not os.path.isfile(config_file):
                print(f"No config file found in: {config_file}")
                continue

            config = ConfigReader(config_file)
            for trial in config.iterate_over_sets():
                yield trial

    def get_subject_iterator(self, mode):
        if mode == "train":
            return self.iterate_over_files(self._train_files)
        elif mode == "test":
            return self.iterate_over_files(self._test_files)
        else:
            raise Exception(f"Unknown mode: {mode}")
