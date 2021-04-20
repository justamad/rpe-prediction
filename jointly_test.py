from src.processing import calculate_magnitude
from jointly.log import logger

import jointly
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

logger.setLevel(10)
matplotlib.use("TkAgg")

files = [625, 626, 627, 628, 629, 630]


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
p_2 = read_csv_and_convert_timestamp("629.csv")

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

extractor = jointly.ShakeExtractor()
synchronizer = jointly.Synchronizer(sources, ref_source_name, extractor)
synced_data = synchronizer.get_synced_data()

a = synced_data['IMU_1']
b = synced_data['IMU_2']
start = max(a.index.min(), b.index.min())
end = min(a.index.max(), b.index.max())
print(start, end)
a = a.loc[(a.index > start) & (a.index <= end)]
b = b.loc[(b.index > start) & (b.index <= end)]
print(a)
print(b)

plt.cla()
plt.clf()
plt.close()
plt.plot(a.index, a['ACCELERATION'], label="IMU 1")
plt.plot(b.index, b['ACCELERATION'], label="IMU 2")
plt.legend()
plt.show()
