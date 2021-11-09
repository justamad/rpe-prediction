from os.path import join

import os
import json
import logging


class LoadingException(Exception):
    pass


class BaseSubjectLoader(object):

    def __init__(self):
        pass

    def get_nr_of_sets(self):
        pass

    def get_trial_by_set_nr(self, trial_nr: int):
        pass


class RPESubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject: str):
        super().__init__()
        json_file = join(root_path, "rpe.json")
        if not os.path.isfile(json_file):
            raise LoadingException(f"RPE file {json_file} does not exists!")

        with open(json_file) as f:
            rpe_values = json.load(f)

        self._trials = {k: v for k, v in enumerate(rpe_values['rpe_ratings'])}
        self._subject = subject

    def get_nr_of_sets(self):
        return len(self._trials)

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial {trial_nr}")
        return self._trials[trial_nr]

    def __repr__(self):
        return f"RPELoader {self._subject}"


class StereoAzureSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject: str):
        super().__init__()
        self._azure_path = join(root_path, "azure")
        if not os.path.exists(self._azure_path):
            raise LoadingException(f"Azure file not present in {self._azure_path}")

        # Load data into two dictionaries: {trial_n: abs_path, ...}
        all_trials = list(map(lambda x: join(self._azure_path, x), os.listdir(self._azure_path)))
        all_trials = list(filter(lambda x: os.path.isdir(x), all_trials))

        self._sub_trials = {int(v[-6:-4]) - 1: v for v in filter(lambda x: 'sub' in x, all_trials)}
        self._master_trials = {int(v[-9:-7]) - 1: v for v in filter(lambda x: 'master' in x, all_trials)}
        self._subject = subject

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._sub_trials or trial_nr not in self._master_trials:
            raise LoadingException(f"{str(self)}: Error when loading set: {trial_nr} for subject {self._azure_path}")
        return join(self._sub_trials[trial_nr], "positions_3d.csv"), \
               join(self._master_trials[trial_nr], "positions_3d.csv")

    def get_nr_of_sets(self):
        return min(len(self._sub_trials), len(self._master_trials))

    def __repr__(self):
        return f"StereoAzureLoader {self._subject}"


class FusedAzureSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject: str):
        super().__init__()
        if not os.path.exists(root_path):
            raise LoadingException(f"Azure file not present in {root_path}")

        # Load data into dictionary: {trial_n: absolute_path, ...}
        files = list(filter(lambda x: 'azure' in x, os.listdir(root_path)))
        self._trials = {int(v.split('_')[0]): join(root_path, v) for v in files}
        self._subject = subject

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self._trials:
            raise LoadingException(f"{str(self)}: Could not load trial: {trial_nr}")
        return self._trials[trial_nr]

    def get_nr_of_sets(self):
        return len(self._trials)

    def __repr__(self):
        return f"FusedAzureLoader {self._subject}"


class SubjectDataCollector(object):

    def __init__(self, subject_root_path: str, data_loaders: dict, subject_name: str, nr_sets: int = 12):
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
