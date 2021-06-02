import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut


class GridSearching(object):

    def __init__(self, model, scaler, parameters, groups, learner_name):
        """
        Constructor for Grid Search class
        @param model: the current regression model
        @param scaler: the current scaler
        @param parameters: the parameter search space
        @param groups: the current groups
        @param learner_name: the name of the learner
        """
        self._steps = [
            # ("remove_constants", constants_remover()),
            ("scaler", scaler),
            # ("feature_selection", selector),
            ("regression", model)
        ]

        # if not skip_balancing:
        #     steps.insert(2, ('balance_sampling', balancer()))

        self._parameters = parameters
        self._groups = groups
        self._learner_name = learner_name

    def perform_grid_search(self, input_data, ground_truth):
        """
        Perform a grid search
        @param input_data: the input training data
        @param ground_truth: ground truth data
        @return: Grid search object with best performing model
        """
        pipe = Pipeline(steps=self._steps)
        logo = LeaveOneGroupOut()

        search = GridSearchCV(estimator=pipe,
                              param_grid=self._parameters,
                              cv=logo.get_n_splits(groups=self._groups),
                              verbose=10)

        print(search)
        search.fit(input_data, ground_truth)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        results = pd.DataFrame(search.cv_results_)
        results.to_csv(f"{self._learner_name}_results.csv", sep=';', index=False)
        return search
