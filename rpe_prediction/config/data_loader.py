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
        return self._trials[trial_nr - 1]


class AzureLoader(BaseLoader):

    def __init__(self, root_path, cam_type="sub"):
        super().__init__()
        self._azure_path = join(root_path, "azure")
        self._cam_type = cam_type
        if not os.path.exists(self._azure_path):
            raise LoadingException(f"Azure file not present in {self._azure_path}")

        self._trials = sorted(filter(lambda x: cam_type in x, os.listdir(self._azure_path)))
        if len(self._trials) == 0:
            raise LoadingException(f"Could not find trials for {cam_type} camera type.")

    def get_trial_by_set_nr(self, trial_nr: int):
        """
        Return azure Kinect data
        # TODO: Implement lookup in cached trials
        @param trial_nr: the current set number, starts at 1
        @return: the current RPE value for given set number
        """
        trial = join(self._azure_path, f"{trial_nr + 1:02}_{self._cam_type}")
        if not os.path.exists(trial):
            raise LoadingException(f"{str(self)}: Could not load trial: {trial}")
        return trial

    def get_nr_of_sets(self):
        return len(self._trials)

    def __repr__(self):
        return "AzureLoader"


data_loaders = {'rpe': RPELoader,
                'azure': AzureLoader}


class DataLoader(object):

    def __init__(self, root_path):
        """
        Create instance of DataLoader for given root path
        @param root_path: current path to subject folder
        """
        self._file_loaders = {}
        for loader_name, loader in data_loaders.items():
            current_loader = loader(root_path)
            self._file_loaders[loader_name] = current_loader

        found_sets = list(map(lambda l: l.get_nr_of_sets(), self._file_loaders.values()))
        result = found_sets.count(found_sets[0]) == len(found_sets)
        if not result:
            print(f"Set(s) are missing for subject: {root_path}")
        self._sets = found_sets[0]

    def iterate_over_sets(self):
        """
        Iterate over the entire set
        @return: Iterator over entire training set
        """
        for current_set in range(self._sets):
            try:
                trial_dic = {k: v.get_trial_by_set_nr(current_set) for k, v in self._file_loaders.items()}
                yield trial_dic
            except LoadingException as e:
                print(e)
