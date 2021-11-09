from .data_loaders import LoadingException

import logging


class SubjectDataCollector(object):

    def __init__(
            self,
            subject_root_path: str,
            data_loaders: dict,
            subject_name: str,
            nr_sets: int = 12
    ):
        self._file_loaders = {}
        for loader_name, loader in data_loaders.items():
            current_loader = loader(subject_root_path, subject_name)
            self._file_loaders[loader_name] = current_loader

        found_sets = list(map(lambda l: l.get_nr_of_sets(), self._file_loaders.values()))
        result = found_sets.count(found_sets[0]) == len(found_sets)
        if not result:
            logging.warning(f"Set(s) are missing for subject: {subject_root_path}")

        self._nr_sets = nr_sets

    def iterate_over_sets(self):
        for current_set in range(self._nr_sets):
            try:
                trial_dic = {k: v.get_trial_by_set_nr(current_set) for k, v in self._file_loaders.items()}
                trial_dic['nr_set'] = current_set
                yield trial_dic
            except LoadingException as e:
                logging.warning(e)
