from .data_collector import SubjectDataCollector
from os.path import join
from typing import List

from src.dataset.data_loaders import (
    LoadingException,
    StereoAzureSubjectLoader,
    RPESubjectLoader,
    ECGSubjectLoader,
    IMUSubjectLoader,
)

import os
import logging
import shutil


class SubjectDataIterator(object):

    STEREO_AZURE = "azure"
    RPE = "rpe"
    ECG = "ecg"
    IMU = "imu"

    loader_names = {
        STEREO_AZURE: StereoAzureSubjectLoader,
        RPE: RPESubjectLoader,
        ECG: ECGSubjectLoader,
        IMU: IMUSubjectLoader,
    }

    def __init__(
            self,
            base_path: str,
            data_loader: List[str],
            dst_path: str = None,
            log_path: str = None,
    ):
        self._base_path = base_path
        self._dst = dst_path
        self._log_path = log_path
        self._data_loaders_dict = {self.loader_names[loader_name]: loader_name for loader_name in data_loader}

    def iterate_over_all_subjects(self):
        return self.iterate_over_specific_subjects()

    def iterate_over_specific_subjects(self, *subjects):
        for subject_id, loader in enumerate(self._load_and_yield_subject_data_collectors(list(subjects))):
            for trial in loader.iterate_over_sets(group_id=subject_id):
                yield trial

    def _load_and_yield_subject_data_collectors(self, subject_list: List[str]):
        cur_dir = os.getcwd()
        subjects = os.listdir(self._base_path)
        if subject_list:
            subjects = list(filter(lambda s: s in subject_list, subjects))

        for subject in subjects:
            try:
                yield SubjectDataCollector(
                    subject_root_path=join(self._base_path, subject),
                    data_loaders=self._data_loaders_dict,
                    subject=subject,
                    nr_sets=12,
                    log_path=self._log_path,
                )
                # shutil.copy(
                #     src=join(self._base_path, subject, "rpe_ratings.json"),
                #     dst=join(self._dst, subject, "rpe_ratings.json"),
                # )

            except LoadingException as e:
                logging.warning(f"Data Loader failed for subject {subject}: {e}")
