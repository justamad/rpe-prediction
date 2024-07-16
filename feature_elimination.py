import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.dataset import (
    extract_dataset_input_output,
    add_rolling_statistics,
    normalize_data_by_subject,
    drop_correlated_features,
)


def eliminate_features(X: pd.DataFrame, y: pd.DataFrame, steps: int, num_features: int):
    timestamp = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    print(X.shape)
    subjects = y["subject"].unique()
    groups = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))
    y = y["rpe"]
    group_k_fold = GroupKFold(n_splits=16)
    cv = group_k_fold.split(X, y, groups)

    rfecv = RFECV(
        estimator=XGBRegressor(),
        step=steps,
        cv=cv,
        scoring='r2',
        verbose=10,
        n_jobs=-1,
        min_features_to_select=num_features,
    )

    rfecv.fit(X, y)
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    print("Number of scores: ", n_scores)
    print(f"Optimal number of features: {rfecv.n_features_}")

    rfe_df = pd.DataFrame(
        data=rfecv.ranking_,
        index=X.columns,
        columns=["Rank"],
    ).sort_values(by="Rank", ascending=True)
    rfe_df.index.names = ["Feature"]
    rfe_df.to_csv(f"{timestamp}_features.csv")

    cv_results_df = pd.DataFrame(rfecv.cv_results_)
    cv_results_df.to_csv("cv_results.csv")

    # Calculate confidence interval bounds
    # confidence_interval = 1.96  # 95% confidence interval
    # upper_bound = rfecv.cv_results_['mean_test_score'] + confidence_interval * rfecv.cv_results_['std_test_score']
    # lower_bound = rfecv.cv_results_['mean_test_score'] - confidence_interval * rfecv.cv_results_['std_test_score']

    plt.figure(figsize=(10, 6))
    plt.errorbar(range(rfecv.min_features_to_select, len(rfecv.cv_results_['mean_test_score']) + 1),
                 rfecv.cv_results_['mean_test_score'],
                 yerr=rfecv.cv_results_['std_test_score'],
                 marker='o', linestyle='', capsize=3)
    plt.title(f'RFECV - Optimal Number of Features: {rfecv.n_features_}')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('R2')
    plt.savefig(f"{timestamp}.png")

    X = X.loc[:, rfecv.support_]
    return X


def pre_filter(X: pd.DataFrame, y: pd.DataFrame, k=600):
    y = y["rpe"]
    # Use SelectKBest to select, for example, the top 5 features based on univariate statistics (f_regression)
    k_best_selector = SelectKBest(f_regression, k=k)
    X_train_k_best = k_best_selector.fit_transform(X, y)
    selected_features_mask = k_best_selector.get_support()
    print("K best features: ", X.columns[selected_features_mask])

    # Now, you can use the selected features to filter out weak features
    X_train_filtered = X.loc[:, selected_features_mask]
    return X_train_filtered


def correlation_filter(X: pd.DataFrame, y: pd.DataFrame, k=300):
    y = y["rpe"]
    correlations = np.abs(X.corrwith(y))
    selected_features_mask = correlations.nlargest(k).index
    X_selected = X[selected_features_mask]
    return X_selected


def variance_threshold(X: pd.DataFrame, threshold=0.01):
    variance_selector = VarianceThreshold(threshold=threshold)
    X_train_high_variance = variance_selector.fit_transform(X)
    X = X.loc[:, variance_selector.get_support()]
    return X


if __name__ == "__main__":
    df_con = pd.read_csv("data/training/concentric_stat.csv", index_col=0)
    df_ecc = pd.read_csv("data/training/eccentric_stat.csv", index_col=0)
    df_full = pd.read_csv("data/training/full_stat.csv", index_col=0)

    for col in ["rpe", "set_id", "subject"]:
        df_ecc.drop(col, axis=1, inplace=True)
        df_con.drop(col, axis=1, inplace=True)

    df_full = pd.concat([df_con, df_ecc, df_full], axis=1)
    X, y = extract_dataset_input_output(df=df_full, labels="rpe")

    sensor = "KINECT"
    print(X.shape)
    drop_columns = [col for col in X.columns if sensor not in col]
    X.drop(columns=drop_columns, inplace=True, errors="ignore")

    X = normalize_data_by_subject(X, y)
    temp_context_df = add_rolling_statistics(X, y, win=[6])
    temp_context_df = normalize_data_by_subject(temp_context_df, y)
    X = pd.concat([X, temp_context_df], axis=1)

    # Trivial feature removal
    X = variance_threshold(X, threshold=0.01)
    X = drop_correlated_features(X, threshold=0.95)
    # X = correlation_filter(X, y, k=50)


    # X = pre_filter(X, y, k=600)
    X = eliminate_features(X, y, steps=10, num_features=1)
    df = pd.concat([X, y], axis=1)
    df.to_csv("imu_features.csv", index=False)
