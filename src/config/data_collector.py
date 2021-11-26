from .data_loaders import LoadingException
from src.utils import create_folder_if_not_already_exists
from os.path import join

import logging


class SubjectDataCollector(object):

    def __init__(
            self,
            subject_root_path: str,
            data_loaders: dict,
            subject_name: str,
            nr_sets: int = 12
    ):
        self._nr_sets = nr_sets
        self._subject_name = subject_name

        self._file_loaders = {}
        for loader_name, loader in data_loaders.items():
            current_loader = loader(subject_root_path, subject_name)
            self._file_loaders[loader_name] = current_loader

    def iterate_over_sets(
            self,
            group_id: int,
            log_path: str = None,
    ):
        for current_set in range(self._nr_sets):
            try:
                trial_dic = {k: v.get_trial_by_set_nr(current_set) for k, v in self._file_loaders.items()}
                trial_dic['nr_set'] = current_set
                trial_dic['group'] = group_id
                trial_dic['subject_name'] = self._subject_name

                if log_path is not None:
                    cur_log_path = join(log_path, self._subject_name, f"{current_set}_set")
                    create_folder_if_not_already_exists(cur_log_path)
                    trial_dic['log_path'] = cur_log_path

                yield trial_dic
            except LoadingException as e:
                logging.warning(e)
