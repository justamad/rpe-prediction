from typing import List
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score
from os.path import join

from .ml_model_config import (
    SVRModelConfig,
    GBRModelConfig,
    SVMModelConfig,
)

import pandas as pd
import logging

metrics = {
    "regression": [
        "r2",
        "neg_mean_squared_error",
        "neg_mean_absolute_error",
        "max_error",
    ],
    "classification": {
        "f1_score": make_scorer(f1_score, average="micro"),
        "precision_score": make_scorer(precision_score, average="micro"),
        "recall_score": make_scorer(recall_score, average="micro"),
        "accuracy_score": make_scorer(accuracy_score),
    }
}

models = {
    "regression":
        [
            SVRModelConfig(),
        ],
    "classification":
        [
            SVMModelConfig(),
        ]
}


class MLOptimization(object):

    def __init__(
            self,
            groups: List,
            task: str,
            mode: str,
            X_train: pd.DataFrame,
            y_train: pd.Series,
    ):
        if task not in ["regression", "classification"]:
            raise AttributeError(f"Unknown ML task given: {task}")

        if mode not in ["grid", "random"]:
            raise AttributeError(f"Unknown ML mode given: {mode}")

        self._groups = groups
        self._task = task
        self._mode = mode
        self._X_train = X_train
        self._y_train = y_train

    def perform_grid_search_with_cv(self, log_path: str):
        for model_config in models[self._task]:
            steps = [
                ("balance_sampling", RandomOverSampler()),
                (str(model_config), model_config.model),
            ]

            parameters = model_config.parameters

            pipe = Pipeline(steps=steps)
            logo = LeaveOneGroupOut()

            if self._mode == "grid":
                grid_search = GridSearchCV(
                    estimator=pipe,
                    param_grid=parameters,
                    cv=logo.get_n_splits(groups=self._groups),
                    n_jobs=-1,
                    verbose=10,
                    scoring=metrics[self._task],
                    error_score="raise",
                    refit="accuracy_score",
                )
            else:
                grid_search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=parameters,
                    n_iter=20,
                    cv=logo.get_n_splits(groups=self._groups),
                    n_jobs=-1,
                    verbose=10,
                    scoring=metrics[self._task],
                    error_score="raise",
                    refit="accuracy_score",
                )

            logging.info(grid_search)
            logging.info(f"Input shape: {self._X_train.shape}")

            grid_search.fit(self._X_train, self._y_train)
            logging.info(f"Best parameter (CV score={grid_search.best_score_:.5f})")
            logging.info(grid_search.best_params_)

            result_df = pd.DataFrame(grid_search.cv_results_)
            result_df = result_df.drop(["params"], axis=1)
            result_df.to_csv(join(log_path, str(model_config) + ".csv"), sep=";")
