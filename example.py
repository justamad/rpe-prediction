from tsfresh.examples import load_robot_execution_failures
from tsfresh import extract_features, select_features

import tsfresh

tsfresh.examples.robot_execution_failures.download_robot_execution_failures()

if __name__ == '__main__':
    df, y = load_robot_execution_failures()
    X_extracted = extract_features(df, column_id='id', column_sort='time')
    X_selected = select_features(X_extracted, y)
