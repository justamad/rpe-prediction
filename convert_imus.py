from src.processing import filter_dataframe

import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

file = "data/sensor-data.parquet"


def read_parquet_sensor_data(file_name: str, timestamp_as_index=True) -> pd.DataFrame:
    df = pd.read_parquet(file_name)
    drop_columns = ["studyId", "userId", "accuracy"]

    timestamp = df["timestamp"].explode(ignore_index=True)
    sensor_timestamp = df["sensorTimestamp"].explode(ignore_index=True)
    studyId = df["studyId"].explode(ignore_index=True)
    userId = df["userId"].explode(ignore_index=True)
    deviceId = df["deviceId"].explode(ignore_index=True)
    type = df["type"].explode(ignore_index=True)
    value = df["value"].explode(ignore_index=True)
    accuracy = df["accuracy"].explode(ignore_index=True)

    df = pd.concat([timestamp, sensor_timestamp, studyId, userId, deviceId, type, value, accuracy], axis="columns")

    if timestamp_as_index:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
        df.set_index("timestamp", inplace=True)

    return df.drop(list(drop_columns), axis="columns")


sensors = {625: "Chest",
           626: "Lumbar Spine",
           627: "Left Leg Upper",
           628: "Right Leg Upper",
           629: "Right Leg Lower",
           630: "Left Leg Lower"
           }


def read_massive_csv_file(file_name):
    data = pd.read_csv(file_name, delimiter=',')
    print(data.shape)
    data = filter_dataframe(data, ['studyId', 'userId', 'accuracy'])


def resort_file(data):
    device_ids = data['deviceId']
    print(f"Found sensors: {pd.unique(device_ids)}")

    for sensor_id, label in sensors.items():
        df = data.loc[data['deviceId'] == sensor_id].copy()
        df = df.reset_index(drop=True).pivot(index='sensorTimestamp', columns='type', values='value')
        df.to_csv(f'{sensor_id}.csv', sep=';')
        print(df)


if __name__ == '__main__':
    df = read_parquet_sensor_data(file)
    print(df.columns)
    # print(df.index)
    resort_file(df)

    # read_massive_csv_file(file)
