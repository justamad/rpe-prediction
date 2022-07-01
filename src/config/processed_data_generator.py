from os.path import join

import pandas as pd
import os


class ProcessedDataGenerator(object):

    def __init__(self, path: str):
        self._path = path
        self._subjects = filter(lambda x: not x.startswith("."), os.listdir(path))

    def generate(self):
        for group_id, subject in enumerate(self._subjects):
            files = os.listdir(join(self._path, subject))
            files = sorted(filter(lambda x: "hrv" in x, files))
            for file in files:
                nr_set = int(file.split("_")[0])

                # azure = pd.read_csv(join(self._path, subject, file), sep=";", index_col="timestamp", parse_dates=True)
                hrv_df = pd.read_csv(join(self._path, subject, file), sep=";", index_col=0, parse_dates=True)
                # azure.index = pd.to_datetime(azure.index)

                dic = {
                    # "azure": azure,
                    "hrv": hrv_df,
                    "subject": subject,
                    "nr_set": nr_set,
                    "rpe": 15,
                    "group": group_id,
                }
                yield dic
