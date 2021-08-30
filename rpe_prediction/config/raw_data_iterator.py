from .data_loaders import SubjectDataCollector, LoadingException, StereoAzureSubjectLoader, RPESubjectLoader, \
    FusedAzureSubjectLoader
from os.path import join

import os
import logging

loader_names = {
    StereoAzureSubjectLoader: 'azure',
    RPESubjectLoader: 'rpe',
    FusedAzureSubjectLoader: 'azure'
}


class SubjectDataIterator(object):

    def __init__(self, base_path):
        """
        Constructor for SubjectDataIterator
        @param base_path: the base path to experiment folder, e.g. ../data/raw/
        """
        self._base_path = base_path
        self._data_loaders_dict = {}

    def add_loader(self, loader):
        """
        Add an data loader to the subject iterator
        @param loader: the new data loader, e.g. azure or RPE loader
        @return: self instance as sort of interface pattern
        """
        loader_name = loader_names[loader]
        self._data_loaders_dict[loader_name] = loader
        return self

    def load_data_collectors(self):
        """
        Function checks for valid configuration files
        @return: a list that contains all configuration files
        """
        data_loaders = {}

        for subject in os.listdir(self._base_path):
            try:
                data_loaders[subject] = SubjectDataCollector(join(self._base_path, subject),
                                                             self._data_loaders_dict,
                                                             subject)
            except LoadingException as e:
                logging.warning(f"Data Loader failed for subject {subject}: {e}")

        return data_loaders

    def iterate_over_all_subjects(self):
        """
        Method returns an iterator over all sets of the entire loaded datasets
        @return: iterator that yields a data dictionary
        """
        subject_data_loaders = self.load_data_collectors()
        logging.info(f"Found {len(subject_data_loaders)} subject folders.")

        for subject_id, (subject_name, data_loader) in enumerate(subject_data_loaders.items()):
            for trial in data_loader.iterate_over_sets():
                trial['group'] = subject_id
                trial['subject_name'] = subject_name
                yield trial

    def iterate_over_specific_subjects(self, *subjects):
        """
        Method returns an iterator for given specific subject
        @param subjects: dynamic list of individual subject names
        @return: iterator over all sets
        """
        subject_data_loaders = self.load_data_collectors()
        logging.info(f"Found {len(subject_data_loaders)} subject folders.")

        for subject_id, subject_name in enumerate(subjects):
            if subject_name not in subject_data_loaders:
                logging.warning(f"Couldn't load data for subject: {subject_name}")
                continue
            for trial in subject_data_loaders[subject_name].iterate_over_sets():
                trial['group'] = subject_id
                trial['subject_name'] = subject_name
                yield trial
