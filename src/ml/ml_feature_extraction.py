import matplotlib.pyplot as plt
import pandas as pd

from typing import Tuple
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import GroupKFold
from os.path import join


def eliminate_features_rfecv(X: pd.DataFrame, y: pd.DataFrame, steps: int, min_features: int, log_path: str):
    subjects = y["subject"].unique()
    groups = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))
    y = y["rpe"]
    group_k_fold = GroupKFold(n_splits=16)
    cv = group_k_fold.split(X, y, groups)

    rfecv = RFECV(
        estimator=XGBRegressor(),
        step=steps,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=10,
        n_jobs=-1,
        min_features_to_select=min_features,
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
    rfe_df.to_csv(join(log_path, f"rfecv.csv"))

    cv_results_df = pd.DataFrame(rfecv.cv_results_)
    cv_results_df.to_csv(join(log_path, "cv_results.csv"))

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
    plt.savefig(join(log_path, "rfecv.png"))

    X = X.loc[:, rfecv.support_]
    return X


def eliminate_features_with_rfe(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        step: int = 10,
        n_features: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selector = RFE(
        estimator=XGBRegressor(),
        n_features_to_select=n_features,
        step=step,
        verbose=10,
    )
    selector.fit(X_train, y_train)
    mask = selector.support_

    rfe_df = pd.DataFrame(
        data=selector.ranking_,
        index=X_train.columns,
        columns=["Rank"],
    ).sort_values(by="Rank", ascending=True)
    rfe_df.index.names = ["Feature"]

    return X_train.loc[:, mask], rfe_df
