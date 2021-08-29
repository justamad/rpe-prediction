from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

import pandas as pd
import logging

scoring = {'R2': 'r2',
           'MSE': 'neg_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}


class GridSearching(object):

    def __init__(self, groups, model, scaler, parameters, learner_name, balancer):
        self._steps = [
            ("scaler", scaler),
            ('balance_sampling', balancer),
            (learner_name, model)
        ]

        self._parameters = parameters
        self._groups = groups
        self._learner_name = learner_name

    def perform_grid_search(self, input_data: pd.DataFrame, ground_truth: pd.DataFrame, output_file: str = None):
        pipe = Pipeline(steps=self._steps)
        logo = LeaveOneGroupOut()

        search = GridSearchCV(estimator=pipe,
                              param_grid=self._parameters,
                              cv=logo.get_n_splits(groups=self._groups),
                              n_jobs=-1,
                              verbose=10,
                              scoring=scoring,
                              refit='MSE')

        logging.info(search)
        search.fit(input_data, ground_truth)
        logging.info("Best parameter (CV score=%0.3f):" % search.best_score_)
        logging.info(search.best_params_)
        results = pd.DataFrame(search.cv_results_)
        results = results.drop(['params'], axis=1)
        results.to_csv(output_file, sep=';', index=False)
        return search
