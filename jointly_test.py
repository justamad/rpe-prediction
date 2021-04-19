from src.processing import calculate_magnitude
from src.jointly import ShakeExtractor, Synchronizer

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("TkAgg")


def read_csv_and_convert_timestamp(file_name):
    df = pd.read_csv(file_name, sep=';', index_col=False)
    df["sensorTimestamp"] = pd.to_datetime(df["sensorTimestamp"], unit="s")
    df = df.set_index("sensorTimestamp", drop=True)
    index = df.index
    # Calculate magnitude
    df = calculate_magnitude(df, "_X")
    df = df.set_index(index)
    return df


p_1 = read_csv_and_convert_timestamp("625.csv")
print(p_1)

p_2 = read_csv_and_convert_timestamp("629.csv")
print(p_1)

sources = {
    'IMU_1': {
        'data': p_1,
        'ref_column': 'ACCELERATION'
    },
    'IMU_2': {
        'data': p_2,
        'ref_column': 'ACCELERATION'}
}

ref_source_name = 'IMU_1'

extractor = ShakeExtractor()
synchronizer = Synchronizer(sources, ref_source_name, extractor)
synced_data = synchronizer.get_synced_data()
