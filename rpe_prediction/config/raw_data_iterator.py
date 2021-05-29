from .data_loader import DataCollector, LoadingException, StereoAzureLoader, RPELoader
from os.path import join

import os


class RawDataLoaderSet(dict):

    def __init__(self):
        super().__init__({'rpe': RPELoader,
                          'azure': StereoAzureLoader})


class KinectFusionLoaderSet(dict):

    def __init__(self):
        super().__init__({'azure': StereoAzureLoader})


class SubjectDataIterator(object):

    def __init__(self, base_path, data_loaders_dict):
        """
        Constructor for SubjectDataIterator
        @param base_path: the base path to experiment folder, e.g. ../data/raw/
        @param data_loaders_dict: a dictionary that defines the data to be loaded
        """
        self._base_path = base_path
        self._data_loaders_dict = data_loaders_dict
        self._subject_data_loaders = self.load_config_files()
        print(f"Found {len(self._subject_data_loaders)} subject folders.")

    def load_config_files(self):
        """
        Function checks for valid configuration files
        @return: a list that contains all configuration files
        """
        data_loaders = {}

        for subject in os.listdir(self._base_path):
            try:
                data_loaders[subject] = DataCollector(join(self._base_path, subject), self._data_loaders_dict)
            except LoadingException as e:
                print(f"Could not load file loader for {subject} trial: {e}")

        return data_loaders

    def iterate_over_all_subjects(self):
        """
        Method returns an iterator over all sets of the entire loaded datasets
        @return: iterator that yields an data dictionary
        """
        for subject_id, (subject_name, data_loader) in enumerate(self._subject_data_loaders.items()):
            for trial in data_loader.iterate_over_sets():
                trial['group'] = subject_id
                yield trial

    def iterate_over_specific_subjects(self, *subjects):
        """
        Method returns an iterator for given specific subject
        @param subjects:
        @return:
        """
        for subject in subjects:
            if subject not in self._subject_data_loaders:
                print(f"Couldn't load data for subject: {subject}")
                continue
            for trial in self._subject_data_loaders[subject].iterate_over_sets():
                yield trial
