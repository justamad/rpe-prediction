from .base_loader import BaseSubjectLoader, LoadingException
from os.path import exists, join
from datetime import datetime
from dateutil.relativedelta import relativedelta
from io import StringIO

import json
import pandas as pd


class HRVSubjectLoader(BaseSubjectLoader):

    def __init__(self, root_path: str, subject_name: str):
        super().__init__(subject_name)

        hrv_file = join(root_path, f"ecg_hrv_30-1.txt")
        if not exists(hrv_file):
            raise LoadingException(f"Given file {hrv_file} does not exist.")

        json_file = join(root_path, "time_selection.json")
        if not exists(json_file):
            raise LoadingException(f"Time selection file {json_file} does not exists.")

        with open(json_file) as json_content:
            data = json.load(json_content)

        self._sets = {index: value for index, value in enumerate(data["non_truncated_selection"]["set_times"])}
        self._df = self.read_time_varying_results(hrv_file)
        # df.index = pd.to_datetime(df.index)

    def get_trial_by_set_nr(self, trial_nr: int) -> pd.DataFrame:
        if trial_nr >= len(self._sets):
            raise LoadingException(f"Couldn't load data for trial {trial_nr}.")

        set_1 = self._sets[trial_nr]
        start_dt = datetime.strptime(set_1["start"], '%H:%M:%S.%f') + relativedelta(years=+70)
        end_dt = datetime.strptime(set_1["end"], '%H:%M:%S.%f') + relativedelta(years=+70)

        sub_df = self._df.loc[(self._df.index > start_dt) & (self._df.index < end_dt)]
        return sub_df

    def get_nr_of_sets(self):
        return len(self._sets)

    @staticmethod
    def read_time_varying_results(path: str) -> pd.DataFrame:
        with open(path, "r") as f:
            lines = f.readlines()[148:]
            result_lines = []
            for line in lines:
                line = ";".join([s.strip() for s in line.split(";")][1:-2])
                result_lines.append(line)
                if line.isspace() or line == "":
                    break

            header_stripped = result_lines[0].split(";")
            units_stripped = result_lines[1].split(";")
            names = [f"{h} {u}".strip() for h, u in zip(header_stripped, units_stripped)]

            df: pd.DataFrame = pd.read_csv(StringIO("\n".join(result_lines[2:])), sep=";", names=names)
            df["time"] = pd.to_datetime(df["Time (hh:mm:ss)"] + " 1970", format="%H:%M:%S %Y")  # , utc=True)
            df = df.reset_index()
            df = df.set_index("time", drop=True)
            return df

    def __repr__(self):
        return f"HRV Loader {self._subject_name}"
