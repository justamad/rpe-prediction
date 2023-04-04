from .ml_model_config import LearningModelBase
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut
from typing import List, Union
from os.path import join

from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    max_error,
    mean_absolute_percentage_error,
)

import pandas as pd
import numpy as np
import logging


metrics = {
    "regression": {
        "r2": make_scorer(r2_score),
        "neg_mean_squared_error": make_scorer(mean_squared_error),
        "neg_mean_absolute_error": make_scorer(mean_absolute_error),
        "mean_absolute_percentage_error": make_scorer(mean_absolute_percentage_error),
    },
    "classification": {
        "f1_score": make_scorer(f1_score, average="micro"),
        "precision_score": make_scorer(precision_score, average="micro"),
        "recall_score": make_scorer(recall_score, average="micro"),
        "accuracy_score": make_scorer(accuracy_score),
    }
}

refit_metrics = {
    "regression": "r2",
    "classification": "f1_score",
}

evaluation_metric = {
    "regression": make_scorer(r2_score),
    "classification": make_scorer(f1_score, average="micro"),
}


class MLOptimization(object):

    def __init__(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: pd.DataFrame,
            balance: bool,
            task: str,
            mode: str,
            ground_truth: str,
            n_groups: int = -1,
    ):
        if task not in ["regression", "classification"]:
            raise AttributeError(f"Unknown ML task given: {task}")

        if mode not in ["grid", "random"]:
            raise AttributeError(f"Unknown ML mode given: {mode}")

        self._X = X

        subjects = y["subject"].unique()
        if n_groups <= 0:
            y["group"] = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))
        else:
            y["group"] = y["subject"].replace({subject: i % n_groups for i, subject in enumerate(subjects)})

        self._y = y
        self._task = task
        self._mode = mode
        self._balance = balance
        self._ground_truth = ground_truth

    def perform_grid_search_with_cv(self, models: List[LearningModelBase], log_path: str, n_jobs: int = -1):
        for model_config in models:

            steps = [
                (str(model_config), model_config.model),
            ]

            if self._balance:
                steps.insert(0, ("balance_sampling", RandomOverSampler()))

            pipe = Pipeline(steps=steps)
            logo = LeaveOneGroupOut()

            X = self._X
            y = self._y

            if self._mode == "grid":
                ml_search = GridSearchCV(
                    estimator=pipe,
                    param_grid=model_config.parameters,
                    cv=logo.get_n_splits(groups=y["group"]),
                    n_jobs=n_jobs,
                    verbose=10,
                    scoring=metrics[self._task],
                    error_score="raise",
                    refit=refit_metrics[self._task],
                )
            else:
                ml_search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=model_config.parameters,
                    n_iter=12,
                    cv=logo.get_n_splits(groups=y["group"]),
                    n_jobs=n_jobs,
                    verbose=10,
                    scoring=metrics[self._task],
                    error_score="raise",
                    refit=refit_metrics[self._task],
                )

            logging.info(ml_search)
            logging.info(f"Input shape: {self._X.shape}")

            ml_search.fit(X, y[self._ground_truth])
            r_df = pd.DataFrame(ml_search.cv_results_)
            r_df = r_df.drop(["params"], axis=1)

            r_df.to_csv(join(log_path, f"model__{str(model_config)}.csv"), index=False)
