from src.devices import AzureKinect
from src.processing import get_joints_as_list, reshape_data_for_ts, reshape_data_from_ts
from tsfresh.feature_extraction import MinimalFCParameters

import pandas as pd
import numpy as np
import tsfresh

settings = MinimalFCParameters()

def read_multiple_trials(n_trials):
    X_final = []
    y_final = []
    for i in range(n_trials):
        azure = AzureKinect(f"data/justin/azure/0{i+1}_sub")
        df = azure._data
        joints = get_joints_as_list(df, " (x) pos")
        joints.remove('t')
        df = reshape_data_for_ts(df, joints)
        df['timestamp'] += 61867844
        ds = pd.Series(data=np.repeat(i, len(df)))
        X_final.append(df)
        y_final.append(ds)

    X_final = pd.concat(X_final, ignore_index=True)
    y_final = pd.concat(y_final, ignore_index=True)
    return X_final, y_final


df = AzureKinect("data/justin/azure/01_sub")._data
joints = get_joints_as_list(df, " (x) pos")
joints.remove('t')

df = reshape_data_for_ts(df, joints)

if __name__ == '__main__':
    extracted_features = tsfresh.extract_features(df, column_id='id', column_sort='timestamp', default_fc_parameters=settings)
    print(extracted_features)
    print(extracted_features.columns)
    print(extracted_features.index)
    df1 = reshape_data_from_ts(extracted_features)
    print(df1)
