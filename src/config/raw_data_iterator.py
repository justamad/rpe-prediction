from .data_collector import SubjectDataCollector
from os.path import join

from src.config.data_loaders import (
    LoadingException,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
    FusedAzureSubjectLoader,
    ECGLoader,
)

import os
import logging

loader_names = {
    StereoAzureSubjectLoader: 'azure',
    RPESubjectLoader: 'rpe',
    FusedAzureSubjectLoader: 'azure',
    ECGLoader: 'ecg',
}


class SubjectDataIterator(object):

    def __init__(self, base_path: str):
        self._base_path = base_path
        self._data_loaders_dict = {}

    def add_loader(self, loader):
        loader_name = loader_names[loader]
        self._data_loaders_dict[loader_name] = loader
        return self

    def iterate_over_all_subjects(self):
        subject_data_loaders = self._load_data_collectors()
        logging.info(f"Found {len(subject_data_loaders)} subject folders.")

        for subject_id, (subject_name, data_loader) in enumerate(subject_data_loaders.items()):
            for trial in data_loader.iterate_over_sets():
                trial['group'] = subject_id
                trial['subject_name'] = subject_name
                yield trial

    def iterate_over_specific_subjects(self, *subjects):
        subject_data_loaders = self._load_data_collectors(list(subjects))
        logging.info(f"Found {len(subject_data_loaders)} subject folders.")

        for subject_id, subject_name in enumerate(subjects):
            if subject_name not in subject_data_loaders:
                logging.warning(f"Couldn't load data for subject: {subject_name}")
                continue
            for trial in subject_data_loaders[subject_name].iterate_over_sets():
                trial['group'] = subject_id
                trial['subject_name'] = subject_name
                yield trial

    def _load_data_collectors(self, subject_list: list = None) -> dict:
        data_loaders = {}
        subjects = os.listdir(self._base_path)

        if subject_list is not None:
            subjects = list(filter(lambda s: s in subject_list, subjects))

        for subject in subjects:
            try:
                data_loaders[subject] = SubjectDataCollector(
                    subject_root_path=join(self._base_path, subject),
                    data_loaders=self._data_loaders_dict,
                    subject_name=subject,
                    nr_sets=12,
                )

            except LoadingException as e:
                logging.warning(f"Data Loader failed for subject {subject}: {e}")

        return data_loaders
