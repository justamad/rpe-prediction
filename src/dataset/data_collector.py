from os.path import exists
from .data_loaders import LoadingException
from os.path import join

import os
import logging


class SubjectDataCollector(object):

    def __init__(
            self,
            subject_root_path: str,
            data_loaders: dict,
            subject: str,
            nr_sets: int = 12,
            dst_path: str = None,
    ):
        self._nr_sets = nr_sets
        self._subject = subject
        self._dst_path = dst_path

        self._file_loaders = {}
        for loader, loader_name in data_loaders.items():
            current_loader = loader(subject_root_path, subject)
            self._file_loaders[loader_name] = current_loader

    def iterate_over_sets(self, group_id: int):
        for current_set in range(self._nr_sets):
            try:
                trial_dic = {k: v.get_trial_by_set_nr(current_set) for k, v in self._file_loaders.items()}
                trial_dic["nr_set"] = current_set
                trial_dic["group"] = group_id
                trial_dic["subject"] = self._subject

                if self._dst_path is not None:
                    cur_dst_path = join(self._dst_path, self._subject, f"{current_set:02d}_set")
                    if not exists(cur_dst_path):
                        os.makedirs(cur_dst_path)
                    trial_dic["dst_path"] = cur_dst_path

                logging.info(f"Loaded set {current_set} for subject {self._subject}")
                yield trial_dic
            except LoadingException as e:
                logging.warning(e)
