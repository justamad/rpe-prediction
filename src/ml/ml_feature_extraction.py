import numpy as np
import pandas as pd

from typing import Tuple, List
from xgboost import XGBRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import GroupKFold
from os.path import join
from sklearn.linear_model import LinearRegression


def eliminate_features_rfecv(
        X: pd.DataFrame,
        y: pd.DataFrame,
        gt: str,
        n_splits: int,
        steps: int,
        min_features: int,
        log_path: str,
):
    subjects = y["subject"].unique()
    groups = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))
    y = y[gt]
    group_k_fold = GroupKFold(n_splits=n_splits)
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
    rfe_df = pd.DataFrame(
        data=rfecv.ranking_,
        index=X.columns,
        columns=["Rank"],
    ).sort_values(by="Rank", ascending=True)
    rfe_df.index.names = ["Feature"]
    rfe_df.to_csv(join(log_path, f"rfecv.csv"))

    cv_results_df = pd.DataFrame(rfecv.cv_results_)
    cv_results_df.index = create_indices(X.shape[1], min_features, steps)
    cv_results_df.to_csv(join(log_path, "cv_results.csv"))
    X = X.loc[:, rfecv.support_]
    return X


def create_indices(max_features: int, min_features: int, step_size: int) -> List[int]:
    countdown_range = list(range(max_features, min_features - 1, -step_size))
    if min_features < countdown_range[-1]:
        countdown_range.append(min_features)
    return countdown_range[::-1]


def eliminate_features_with_rfe(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        step: int = 10,
        n_features: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    selector = RFE(
        # estimator=XGBRegressor(),
        estimator=LinearRegression(),
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
