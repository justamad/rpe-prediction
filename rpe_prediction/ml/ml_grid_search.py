from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

import pandas as pd
import logging

scoring = ['r2',
           'neg_mean_squared_error',
           'neg_mean_absolute_error',
           'max_error',
           ]


class GridSearching(object):

    def __init__(self, groups: list, model: None, parameters, learner_name: str, balancer):
        self._steps = [
            ('balance_sampling', balancer),
            (learner_name, model)
        ]

        self._parameters = parameters
        self._groups = groups
        self._learner_name = learner_name

    def perform_grid_search_with_cv(self, X_train: pd.DataFrame, y_train: pd.DataFrame, output_file: str = None):
        pipe = Pipeline(steps=self._steps)
        logo = LeaveOneGroupOut()

        search = GridSearchCV(
            estimator=pipe,
            param_grid=self._parameters,
            cv=logo.get_n_splits(groups=self._groups),
            n_jobs=-1,
            verbose=10,
            scoring=scoring,
            error_score='raise',
            refit='r2',
        )

        logging.info(search)
        logging.info(f"Input shape: {X_train.shape}")
        search.fit(X_train, y_train)
        logging.info(f"Best parameter (CV score={search.best_score_:.5f}")
        logging.info(search.best_params_)
        results = pd.DataFrame(search.cv_results_)
        results = results.drop(['params'], axis=1)
        results.to_csv(output_file, sep=';', index=False)
        return search
