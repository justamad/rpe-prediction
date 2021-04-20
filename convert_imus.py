from src.config import read_parquet_sensor_data, resort_file
from src.processing import calculate_and_append_magnitude
from jointly.log import logger
from os.path import join

import jointly
import matplotlib
import matplotlib.pyplot as plt
import argparse

logger.setLevel(10)
matplotlib.use("TkAgg")

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, dest='path', default="data/arne_flywheel")
args = parser.parse_args()

sensors = [625, 626, 627, 628, 629, 630]

df = read_parquet_sensor_data(join(args.path, "sensor-data.parquet"))
sensor_files = resort_file(df, sensors)

sources = {str(k): {'data': calculate_and_append_magnitude(v), 'ref_column': 'MAGNITUDE'} for k, v in sensor_files.items()}
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
