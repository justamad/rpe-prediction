import pandas as pd
import numpy as np
import logging

from .ml_model_config import LearningModelBase
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold
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
            labels: Union[str, List[str]],
            task: str = "regression",
            mode: str = "grid",
            n_splits: int = None,
    ):
        if task not in ["regression", "classification"]:
            raise AttributeError(f"Unknown ML task given: {task}")

        if mode not in ["grid", "random"]:
            raise AttributeError(f"Unknown ML mode given: {mode}")

        if "subject" not in y.columns:
            raise AttributeError("No subject column found in y")

        subjects = y["subject"].unique()
        y["group"] = y["subject"].replace(dict(zip(subjects, range(len(subjects)))))

        if n_splits is None:
            n_splits = len(subjects)
        self._n_splits = n_splits

        self._X = X
        self._y = y
        self._task = task
        self._mode = mode
        self._balance = balance
        self._ground_truth = labels

    def perform_grid_search_with_cv(
            self, models: Union[LearningModelBase, List[LearningModelBase]],
            log_path: str,
            n_jobs: int = -1,
            verbose: int = 10,
    ):
        X = self._X
        if isinstance(X, pd.DataFrame):
            X = X.values
        y = self._y[self._ground_truth].values
        groups = self._y["group"].values

        if not isinstance(models, list):
            models = [models]

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
                    estimator=model_config.model,
                    param_grid=model_config.parameters,
                    cv=group_k_fold.split(X, y, groups),
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
                    cv=group_k_fold.split(X, y, groups),
                    verbose=verbose,
                    scoring=metrics[self._task],
                    refit=refit_metrics[self._task],
                )

            logging.info(ml_search)
            logging.info(f"Input shape: {self._X.shape}")

            ml_search.fit(X, y, X_val=X, y_val=y)
            r_df = pd.DataFrame(ml_search.cv_results_)
            r_df = r_df.drop(["params"], axis=1)

            r_df.to_csv(join(log_path, f"model__{str(model_config)}.csv"), index=False)

    def evaluate_model(
            self, model,
            norm_labels: bool,
            label_mean: Union[float, List[float]],
            label_std: Union[float, List[float]],
    ) -> pd.DataFrame:
        X = self._X
        if isinstance(X, pd.DataFrame):
            X = X.values

        y = self._y[self._ground_truth].values
        groups = self._y["group"].values
        sets = self._y["set_id"].values
        subjects = self._y["subject"].values

        result_df = pd.DataFrame()
        group_k_fold = GroupKFold(n_splits=self._n_splits)
        for train_idx, test_idx in tqdm(group_k_fold.split(X, y, groups)):
            results = {"set_id": sets[test_idx], "subject": subjects[test_idx]}

            if self._balance:
                model = Pipeline(steps=[
                    ("balance_sampling", RandomOverSampler()),
                    ("learner", model),
                ])

            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]

            y_pred = model.fit(X_train, y_train).predict(X_test)

            if norm_labels:
                y_test = y_test * label_std + label_mean
                y_pred = y_pred * label_std + label_mean

            if isinstance(self._ground_truth, list):
                for idx, label_name in enumerate(self._ground_truth):
                    results[label_name + "_prediction"] = y_pred[:, idx]
                    results[label_name + "_ground_truth"] = y_test[:, idx]
            else:
                results["prediction"] = y_pred
                results["ground_truth"] = y_test

            result_df = pd.concat([result_df, pd.DataFrame(results)])

        return result_df
