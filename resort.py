from src.devices import AzureKinect
from src.processing import get_joints_as_list, reshape_data_for_ts
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table


import pandas as pd
import numpy as np
import tsfresh

settings = ComprehensiveFCParameters()


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


X, y = read_multiple_trials(2)

if __name__ == '__main__':
    # extracted_features = tsfresh.extract_features(X, column_id='id', column_sort='timestamp', default_fc_parameters=settings)
    test = tsfresh.select_features(X, y, ml_task='auto')
    # relevance_table = calculate_relevance_table(extracted_features, y, ml_task='auto')
    print(test)
