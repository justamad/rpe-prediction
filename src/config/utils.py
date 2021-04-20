import pandas as pd


def read_parquet_sensor_data(file_name: str, timestamp_as_index=True) -> pd.DataFrame:
    """
    Read a parquet file from SensorHub that contains the IMU data
    @param file_name: full file name to parquet file
    @param timestamp_as_index: flag if timestamps should be the index
    @return: data frame that contains the sensor data organized in rows
    """
    df = pd.read_parquet(file_name)
    drop_columns = ["studyId", "userId", "accuracy", "timestamp"]

    timestamp = df["timestamp"].explode(ignore_index=True)
    sensor_timestamp = df["sensorTimestamp"].explode(ignore_index=True)
    study_id = df["studyId"].explode(ignore_index=True)
    user_id = df["userId"].explode(ignore_index=True)
    device_id = df["deviceId"].explode(ignore_index=True)
    type_field = df["type"].explode(ignore_index=True)
    value = df["value"].explode(ignore_index=True)
    accuracy = df["accuracy"].explode(ignore_index=True)

    df = pd.concat([timestamp, sensor_timestamp, study_id, user_id, device_id, type_field, value, accuracy], axis="columns")

    if timestamp_as_index:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
        df.set_index("timestamp", inplace=True)

    return df.drop(list(drop_columns), axis="columns")


def resort_file(data, sensors):
    """
    Resort a parquet data file from SensorHub
    @param data: the full data frame
    @param sensors: a list with all sensor IDs (taken from SensorHub)
    @return: dictionary that contains data frame for all files
    """
    device_ids = data['deviceId']
    print(f"Found sensor IDs: {pd.unique(device_ids)}")

    sensor_dict = {}

    for sensor_id in sensors:
        df = data.loc[data['deviceId'] == sensor_id].copy()
        df = df.reset_index(drop=True).pivot(index='sensorTimestamp', columns='type', values='value')
        df["sensorTimestamp"] = pd.to_datetime(df.index, unit="s")
        df = df.set_index("sensorTimestamp", drop=True)
        sensor_dict[sensor_id] = df
        print(df)
        print(df.columns)
        print(df.index)

    return sensor_dict
