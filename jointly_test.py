from src.processing import calculate_magnitude

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import jointly
import pandas as pd
import datetime


def read_csv_and_convert_timestamp(file_name):
    df = pd.read_csv(file_name, sep=';', index_col=False)
    index = pd.DatetimeIndex([datetime.datetime.fromtimestamp(t) for t in df['sensorTimestamp']])
    df = df.drop(['sensorTimestamp'], axis=1).set_index(index)

    # Calculate magnitude
    m = calculate_magnitude(df, "_X")
    m = m.set_index(index)
    return m


p_1 = read_csv_and_convert_timestamp("625.csv")
print(p_1)
plt.plot(p_1['ACCELERATION'])
plt.show()

exit(-1)
p_2 = read_csv_and_convert_timestamp("629.csv")
print(p_1)

sources = {
    'IMU_1': {
        'data': p_1,
        'ref_column': 'ACCELERATION_X'
    },
    'IMU_2': {
        'data': p_2,
        'ref_column': 'ACCELERATION_X'}
}

ref_source_name = 'IMU_1'

extractor = jointly.ShakeExtractor()
synchronizer = jointly.Synchronizer(sources, ref_source_name, extractor)
synced_data = synchronizer.get_synced_data()
