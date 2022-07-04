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
            X: pd.DataFrame,
            y: pd.Series,
            task: str,
            mode: str,
    ):
        if task not in ["regression", "classification"]:
            raise AttributeError(f"Unknown ML task given: {task}")

        if mode not in ["grid", "random"]:
            raise AttributeError(f"Unknown ML mode given: {mode}")

        self._X = X
        self._y = y
        self._task = task
        self._mode = mode

    def perform_grid_search_with_cv(self, log_path: str):
        for model_config in models[self._task]:
            result_df = pd.DataFrame()

            for subject in self._y["subject"].unique():
                logging.info(f"Leave out subject: {subject}")
                mask = self._y["subject"] == subject
                X_train, y_train = self._X[~mask], self._y[~mask]
                X_test, y_test = self._X[mask], self._y[mask]

                steps = [
                    ("balance_sampling", RandomOverSampler()),
                    (str(model_config), model_config.model),
                ]

                pipe = Pipeline(steps=steps)
                logo = LeaveOneGroupOut()

                if self._mode == "grid":
                    grid_search = GridSearchCV(
                        estimator=pipe,
                        param_grid=model_config.parameters,
                        cv=logo.get_n_splits(groups=y_train["group"]),
                        n_jobs=-1,
                        verbose=10,
                        scoring=metrics[self._task],
                        error_score="raise",
                        refit="accuracy_score",
                    )
                else:
                    grid_search = RandomizedSearchCV(
                        estimator=pipe,
                        param_distributions=model_config.parameters,
                        n_iter=20,
                        cv=logo.get_n_splits(groups=y_train["group"]),
                        n_jobs=-1,
                        verbose=10,
                        scoring=metrics[self._task],
                        error_score="raise",
                        refit="accuracy_score",
                    )

                logging.info(grid_search)
                logging.info(f"Input shape: {X_train.shape}")

                grid_search.fit(X_train, y_train["rpe"])
                logging.info(f"Best CV score: {grid_search.best_score_:.5f}, achieved by {grid_search.best_params_}")
                y_pred = grid_search.predict(X_test)
                subject_accuracy = accuracy_score(y_test["rpe"], y_pred)
                logging.info(f"Test subject {subject}, accuracy: {subject_accuracy:.5f}")

                r_df = pd.DataFrame(grid_search.cv_results_)
                r_df = r_df.drop(["params"], axis=1)
                r_df["test_subject"] = subject
                r_df["test_score"] = subject_accuracy
                result_df = pd.concat([result_df, r_df], axis=1)

            result_df.to_csv(join(log_path, str(model_config) + ".csv"), sep=";")
