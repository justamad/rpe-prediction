from tsfresh.examples import load_robot_execution_failures
from tsfresh.utilities.dataframe_functions import check_for_nans_in_columns, impute
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh import extract_features, select_features

import tsfresh

tsfresh.examples.robot_execution_failures.download_robot_execution_failures()

if __name__ == '__main__':
    df, y = load_robot_execution_failures()
    X_extracted = extract_features(df, column_id='id', column_sort='time')
    X_imputed = impute(X_extracted)
    check_for_nans_in_columns(X_imputed)
    X_selected = select_features(X_imputed, y)
    table = calculate_relevance_table(X_imputed, y, ml_task='auto')
    print(table)
