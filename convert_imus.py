from src.processing import filter_dataframe

import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

file = "data/sensor-data.csv"

sensors = {625: "Chest",
           626: "Lumbar Spine",
           627: "Left Leg Upper",
           628: "Right Leg Upper",
           629: "Right Leg Lower",
           630: "Left Leg Lower"
           }


def read_massive_csv_file(file_name):
    data = pd.read_csv(file_name, delimiter=',')
    data = filter_dataframe(data, ['studyId', 'userId', 'accuracy'])
    device_ids = data['deviceId']
    print(f"Found sensors: {pd.unique(device_ids)}")

    for sensor_id, label in sensors.items():
        df = data.loc[data['deviceId'] == sensor_id].copy()
        df = df.reset_index(drop=True).pivot(index='sensorTimestamp', columns='type', values='value')
        df.to_csv(f'{sensor_id}.csv', sep=';')
        print(df)

        # plt.plot(df["ACCELERATION_Y"], label=label)

    # plt.legend()
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    read_massive_csv_file(file)
