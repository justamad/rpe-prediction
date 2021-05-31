from os.path import join

import os
import json


class LoadingException(Exception):
    pass


class BaseLoader(object):

    def __init__(self):
        pass

    def get_nr_of_sets(self):
        pass

    def get_trial_by_set_nr(self, trial_nr: int):
        pass


class RPELoader(BaseLoader):

    def __init__(self, root_path):
        super().__init__()
        json_file = join(root_path, "rpe.json")
        if not os.path.isfile(json_file):
            raise LoadingException(f"RPE file {json_file} does not exists!")

        with open(json_file) as f:
            rpe_values = json.load(f)

        self._trials = rpe_values['rpe_ratings']

    def get_nr_of_sets(self):
        return len(self._trials)

    def get_trial_by_set_nr(self, trial_nr: int):
        """
        Return an RPE value for given set
        @param trial_nr: the current set number, starts at 1
        @return: the current RPE value for given set number
        """
        return self._trials[trial_nr]


class StereoAzureLoader(BaseLoader):

    def __init__(self, root_path):
        super().__init__()
        self._azure_path = join(root_path, "azure")
        if not os.path.exists(self._azure_path):
            raise LoadingException(f"Azure file not present in {self._azure_path}")

        # Load data into two dictionaries: {trial_n: abs_path, ...}
        all_trials = list(map(lambda x: join(self._azure_path, x), os.listdir(self._azure_path)))
        all_trials = list(filter(lambda x: os.path.isdir(x), all_trials))

        self._sub_trials = {int(v[-6:-4]) - 1: v for v in filter(lambda x: 'sub' in x, all_trials)}
        self._master_trials = {int(v[-9:-7]) - 1: v for v in filter(lambda x: 'master' in x, all_trials)}

        if self._sub_trials.keys() != self._master_trials.keys():
            raise LoadingException(f"Trials found: sub={len(self._sub_trials)} master={len(self._master_trials)}")

    def get_trial_by_set_nr(self, trial_nr: int):
        """
        Return azure Kinect data
        @param trial_nr: the current set number, starts at 1
        @return: the current RPE value for given set number
        """
        if trial_nr not in self._sub_trials or trial_nr not in self._master_trials:
            raise LoadingException(f"{str(self)}: Error when loading set: {trial_nr}")
        return self._sub_trials[trial_nr], self._master_trials[trial_nr]

    def get_nr_of_sets(self):
        return min(len(self._sub_trials), len(self._master_trials))

    def __repr__(self):
        return "StereoAzureLoader"


class FusedAzureLoader(BaseLoader):

    def __init__(self, root_path):
        super().__init__()
        if not os.path.exists(root_path):
            raise LoadingException(f"Azure file not present in {root_path}")

        # Load data into dictionary: {trial_n: absolute_path, ...}
        files = list(filter(lambda x: 'azure' in x, os.listdir(root_path)))
        self.trials = {int(v.split('_')[0]): join(root_path, v) for v in files}

    def get_trial_by_set_nr(self, trial_nr: int):
        if trial_nr not in self.trials:
            raise LoadingException(f"{str(self)}: Could not load trial: {trial_nr}")
        return self.trials[trial_nr]

    def get_nr_of_sets(self):
        return len(self.trials)

    def __repr__(self):
        return "FusedAzureLoader"


class DataCollector(object):

    def __init__(self, subject_root_path, data_loaders):
        """
        Constructor for data collector that crawls and prepares all data
        @param subject_root_path: the current path to subject folder
        @param data_loaders: a dictionary that defines the data to be loaded
        """
        self._file_loaders = {}
        for loader_name, loader in data_loaders.items():
            current_loader = loader(subject_root_path)
            self._file_loaders[loader_name] = current_loader

        found_sets = list(map(lambda l: l.get_nr_of_sets(), self._file_loaders.values()))
        result = found_sets.count(found_sets[0]) == len(found_sets)
        if not result:
            print(f"Set(s) are missing for subject: {subject_root_path}")
        self._sets = found_sets[0]

    def iterate_over_sets(self):
        """
        Iterate over the entire set and collect data from individual trials
        @return: Iterator over entire training set
        """
        for current_set in range(self._sets):
            try:
                trial_dic = {k: v.get_trial_by_set_nr(current_set) for k, v in self._file_loaders.items()}
                trial_dic['nr_set'] = current_set
                yield trial_dic
            except LoadingException as e:
                print(e)
