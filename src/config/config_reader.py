from typing import Any

import json
import os


class ConfigReader:

    def __init__(self, trial_configuration_file_name: str):
        if not os.path.exists(trial_configuration_file_name):
            raise Exception(f"Given trial configuration file does not exist: {trial_configuration_file_name}")

        with open(trial_configuration_file_name) as f:
            self.trial_config = json.load(f)

    def get_start_indices_for_set(self, sensor: str, set_nr: int):
        set_list = self.trial_config['sets']
        cur_set = set_list[set_nr]
        cur_sensor = cur_set[sensor]
        return cur_sensor['start_idx'], cur_sensor['end_idx']

    def get_rpe_value_for_set(self, set_nr: int):
        set_list = self.trial_config['sets']
        cur_set = set_list[set_nr]
        return cur_set['rpe']

    def get_synchronization_sensor(self):
        return self.trial_config['sync_sensor']

    def get_all_rpe_values(self):
        set_list = self.trial_config['sets']
        rpe_values = [entry['rpe'] for entry in set_list]
        return rpe_values

    def getboolean(self, name: str) -> bool:
        config = self.trial_config
        return config[name]

    def get(self, name: str) -> Any:
        config = self.trial_config
        return config[name]

    def iterate_over_trials(self):
        for i in range(21):
            yield {'gaitup': self.get_start_indices_for_set('gaitup', i),
                   'faros': self.get_start_indices_for_set('faros', i),
                   'rpe': self.get_rpe_value_for_set(i)}
