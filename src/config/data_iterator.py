from .data_collector import SubjectDataCollector
from os.path import join
from typing import List

from src.config.data_loaders import (
    LoadingException,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
    ECGSubjectLoader,
    IMUSubjectLoader,
)

import os
import logging

loader_names = {
    StereoAzureSubjectLoader: "azure",
    RPESubjectLoader: "rpe",
    ECGSubjectLoader: "ecg",
    IMUSubjectLoader: "imu",
}


class SubjectDataIterator(object):

    def __init__(
            self,
            base_path: str,
            loaders: List,
            log_path: str = None,
    ):
        self._base_path = base_path
        self._log_path = log_path
        self._data_loaders_dict = {loader_names[loader_type]: loader_type for loader_type in loaders}

    def iterate_over_all_subjects(self):
        return self.iterate_over_specific_subjects()

    def iterate_over_specific_subjects(self, *subjects):
        for subject_id, loader in enumerate(self._load_subject_data_collectors(list(subjects))):
            for trial in loader.iterate_over_sets(log_path=self._log_path, group_id=subject_id):
                yield trial

    def _load_subject_data_collectors(self, subject_list: List):
        subjects = os.listdir(self._base_path)
        if subject_list:
            subjects = list(filter(lambda s: s in subject_list, subjects))

        for subject in subjects:
            try:
                yield SubjectDataCollector(
                    subject_root_path=join(self._base_path, subject),
                    data_loaders=self._data_loaders_dict,
                    subject_name=subject,
                    nr_sets=12,
                )

            except LoadingException as e:
                logging.warning(f"Data Loader failed for subject {subject}: {e}")
