from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

import pandas as pd

scoring = {'R2': 'r2',
           'MSE': 'neg_mean_squared_error',
           'MAE': 'neg_mean_absolute_error'}


class GridSearching(object):

    def __init__(self, model, scaler, parameters, groups, learner_name, balancer, selector, constant_remover):
        """
        Constructor for Grid Search class
        @param model: the current regression model to be optimized
        @param scaler: the current scaler for input data
        @param parameters: the parameter search space
        @param groups: the current groups for cross-validation
        @param learner_name: the name of the learner
        """
        self._steps = [
            ("remove_constants", constant_remover),
            ("scaler", scaler),
            ("feature_selection", selector),
            ('balance_sampling', balancer),
            ("regression", model)
        ]

        self._parameters = parameters
        self._groups = groups
        self._learner_name = learner_name

    def perform_grid_search(self, input_data, ground_truth):
        """
        Perform a grid search on the given input data and ground truth data
        @param input_data: the input training data
        @param ground_truth: ground truth data
        @return: Grid search object with best performing model
        """
        pipe = Pipeline(steps=self._steps)
        logo = LeaveOneGroupOut()

        search = GridSearchCV(estimator=pipe,
                              param_grid=self._parameters,
                              cv=logo.get_n_splits(groups=self._groups),
                              n_jobs=-1,
                              verbose=10,
                              scoring=scoring,
                              refit='MSE')

        print(search)
        search.fit(input_data, ground_truth)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        results = pd.DataFrame(search.cv_results_)
        results = results.drop(['params'], axis="columns", inplace=True)
        results.to_csv(f"{self._learner_name}_results.csv", sep=';', index=False)
        return search
