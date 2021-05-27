from .data_loader import DataCollector, LoadingException, StereoAzureLoader, RPELoader, FusionAzureLoader
from os.path import join

import os


class RawDataLoaderSet(dict):

    def __init__(self):
        super().__init__({'rpe': RPELoader,
                          'azure': StereoAzureLoader})


class KinectFusionLoaderSet(dict):

    def __init__(self):
        super().__init__({'azure': FusionAzureLoader})


class SubjectDataIterator(object):

    def __init__(self, input_path, data_loaders_dict):
        self._input_path = input_path
        self._subject_data_loaders = self.load_config_files(data_loaders_dict)
        print(f"Found {len(self._subject_data_loaders)} subject folders.")

    def load_config_files(self, data_loaders_dict):
        """
        Function checks for valid configuration files
        @return: a list that contains all configuration files
        """
        data_loaders = []

        for subject in os.listdir(self._input_path):
            try:
                loader = DataCollector(join(self._input_path, subject), data_loaders_dict)
                data_loaders.append(loader)
            except LoadingException as e:
                print(f"Could not load file loader for {subject} trial: {e}")

        return data_loaders

    def iterate_over_all_subjects(self):
        """
        Method returns an iterator over all sets of the entire loaded datasets
        @return: iterator that yields an data dictionary
        """
        for subject_id, data_loader in enumerate(self._subject_data_loaders):
            for trial in data_loader.iterate_over_sets():
                trial['group'] = subject_id
                yield trial
