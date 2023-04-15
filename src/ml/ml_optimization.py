from .ml_model_config import LearningModelBase
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, LeaveOneGroupOut, GroupKFold
from scipy.stats import pearsonr
from typing import List, Union
from os.path import join
from tqdm import tqdm

from sklearn.metrics import (
    make_scorer,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
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
            n_splits: int = None,
    ):
        if task not in ["regression", "classification"]:
            raise AttributeError(f"Unknown ML task given: {task}")

        if mode not in ["grid", "random"]:
            raise AttributeError(f"Unknown ML mode given: {mode}")

        self._X = X

        subjects = y["subject"].unique()
        y["group"] = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))

        if n_splits is None:
            n_splits = len(subjects)
        self._n_splits = n_splits

        self._y = y
        self._task = task
        self._mode = mode
        self._balance = balance
        self._ground_truth = ground_truth

    def perform_grid_search_with_cv(
            self, models: List[LearningModelBase],
            log_path: str,
            n_jobs: int = -1,
            verbose: int = 1,
    ):
        X = self._X
        y = self._y

        for model_config in models:
            steps = [
                (str(model_config), model_config.model),
            ]

            if self._balance:
                steps.insert(0, ("balance_sampling", RandomOverSampler()))

            pipe = Pipeline(steps=steps)
            group_k_fold = GroupKFold(n_splits=self._n_splits)

            if self._mode == "grid":
                ml_search = GridSearchCV(
                    estimator=pipe,
                    param_grid=model_config.parameters,
                    cv=group_k_fold.split(X, y, y["group"]),
                    n_jobs=n_jobs,
                    verbose=verbose,
                    scoring=metrics[self._task],
                    refit=refit_metrics[self._task],
                )
            else:
                ml_search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=model_config.parameters,
                    n_iter=12,
                    n_jobs=n_jobs,
                    cv=group_k_fold.split(X, y, y["group"]),
                    verbose=verbose,
                    scoring=metrics[self._task],
                    refit=refit_metrics[self._task],
                )

            logging.info(ml_search)
            logging.info(f"Input shape: {self._X.shape}")

            ml_search.fit(X, y[self._ground_truth])
            r_df = pd.DataFrame(ml_search.cv_results_)
            r_df = r_df.drop(["params"], axis=1)

            r_df.to_csv(join(log_path, f"model__{str(model_config)}.csv"), index=False)

    def evaluate_model(self, model, norm_labels: bool, label_mean: float, label_std: float):
        metrics = {
            "r2": r2_score,
            "mse": mean_squared_error,
            "mae": mean_absolute_percentage_error,
            "mape": mean_absolute_percentage_error,
            # "pcc": pearsonr,
        }

        tests = {metric: [] for metric in metrics.keys()}
        result_df = pd.DataFrame()
        for test_group in tqdm(self._y["group"].unique()):
            if self._balance:
                model = Pipeline(steps=[
                    ("balance_sampling", RandomOverSampler()),
                    # ("balance_sampling", SMOTE()),
                    ("learner", model),
                ])

            X_train = self._X.loc[self._y["group"] != test_group, :]
            y_train = self._y.loc[self._y["group"] != test_group, :]
            X_test = self._X.loc[self._y["group"] == test_group, :]
            y_test = self._y.loc[self._y["group"] == test_group, :]

            ground_truth = y_test.loc[:, self._ground_truth].values
            predictions = model.fit(X_train, y_train[self._ground_truth]).predict(X_test)

            if norm_labels == "global":
                ground_truth = ground_truth * label_std + label_mean
                predictions = predictions * label_std + label_mean

            # Calculate all error metrics
            for metric_name, metric in metrics.items():
                tests[metric_name].append(metric(ground_truth, predictions))

            df = pd.DataFrame({
                "ground_truth": ground_truth,
                "predictions": predictions,
                "set_id": y_test["set_id"],
                "subject": y_test["subject"],
            })
            result_df = pd.concat([result_df, df])

        res = {metric: f"${np.mean(values):.2f} \\pm {np.std(values):.2f}$" for metric, values in tests.items()}
        return res, result_df
