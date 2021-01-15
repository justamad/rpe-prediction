import pandas as pd
import os


class AzureKinect(object):

    def __init__(self, data_path):
        if isinstance(data_path, pd.DataFrame):
            self.data = data_path
        elif os.path.exists(data_path):
            self.data = pd.read_csv(data_path, delimiter=';')
            print(self.data)
        else:
            raise Exception(f"Path {data_path} does not exists!")

    def process_raw_data(self, sampling_rate=30):
        # Remove the confidence values and body idx from data frame
        self.data = self.data.loc[:, ~self.data.columns.str.contains('(c)')].copy()
        self.data = self.data.loc[:, ~self.data.columns.str.contains('body_idx')]
        # Convert timestamp to seconds and upsample data
        self.data.loc[:, self.data.columns == 'timestamp'] *= 1e-6

        # Remove timestamp column from data frame
        self.data = self.data.loc[:, ~self.data.columns.str.contains('timestamp')]

    def get_data(self, with_timestamps=False):
        """
        Return the processed skeleton data without timestamps
        :return: Nd-array that contains data with or without timestamps
        """
        if with_timestamps:
            if 'timestamp' not in self.data:
                raise Exception(f"Data for {self} does not contain any timestamps.")
            return self.data.to_numpy()

        if 'timestamp' not in self.data:
            return self.data.to_numpy()
        return self.data.to_numpy()[:, 1:]

    def __repr__(self):
        return "azure"
