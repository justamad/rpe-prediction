from rpe_prediction.config import read_parquet_sensor_data, resort_file
from rpe_prediction.processing import calculate_and_append_magnitude
from jointly.log import logger
from os.path import join

import jointly
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse

logger.setLevel(10)
matplotlib.use("TkAgg")

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, dest='path', default="data/arne_flywheel")
parser.add_argument('--from_scratch', type=bool, dest='from_scratch', default=False)
args = parser.parse_args()

sensors = [625, 626, 627, 628, 629, 630]

if args.from_scratch:
    df = read_parquet_sensor_data(join(args.path, "sensor-data.parquet"), timestamp_as_index=False)
    sensor_files = resort_file(df, sensors)
    sensor_files = {k: calculate_and_append_magnitude(v) for k, v in sensor_files.items()}

    # Save extracted sensor files
    for sensor, data in sensor_files.items():
        data.to_csv(join(args.path, f"{sensor}.csv"), sep=';')
else:
    sensor_files = {}
    for sensor in sensors:
        df = pd.read_csv(join(args.path, f"{sensor}.csv"), sep=';')
        df["sensorTimestamp"] = pd.to_datetime(df["sensorTimestamp"])
        df = df.set_index("sensorTimestamp", drop=True)
        print(df.index)
        sensor_files[sensor] = df

sources = {str(k): {'data': v, 'ref_column': 'MAGNITUDE'} for k, v in sensor_files.items()}
ref_source_name = '625'

extractor = jointly.ShakeExtractor()
synchronizer = jointly.Synchronizer(sources, ref_source_name, extractor)
synced_data = synchronizer.get_synced_data()

plt.close('all')
for label, data in synced_data.items():
    plt.plot(data.index, data['MAGNITUDE'], label=label)
    data.to_csv(join(args.path, f"{label}.csv"), sep=';')

plt.legend()
plt.show()
