from xgboost import XGBRegressor
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


class LearningModelBase(object):

    def __init__(self, model, grid_search_params: dict):
        self.__model = model
        self.__grid_search_params = grid_search_params
        self.__balancer = RandomOverSampler()

    def get_trial_data_dict(self):
        return {
            'model': self.__model,
            'parameters': self.__grid_search_params,
            'learner_name': str(self),
            'balancer': self.__balancer,
        }


class SVRModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {f'{str(self)}__kernel': ('linear', 'rbf'),
                            f'{str(self)}__gamma': [1e-3, 1e-4],
                            f'{str(self)}__C': [1e0, 1e1, 1e2, 1e3], }

        model = SVR()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "svr"


class KNNModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {f'{str(self)}__n_neighbors': [5, 10, 15],
                            f'{str(self)}__weights': ['uniform', 'distance'],
                            f'{str(self)}__leaf_size': [10, 30, 60, 120]}

        model = KNeighborsRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "knn"


class RFModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {f'{str(self)}__n_estimators': [50, 100, 150, 200],
                            f'{str(self)}__criterion': ['mse', 'mae']}

        model = RandomForestRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "rf"


class GBRModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {f'{str(self)}__n_estimators': [15, 50, 150, 500],
                            f'{str(self)}__learning_rate': [0.05, 0.1, 0.2],
                            f'{str(self)}__loss': ["ls", "huber", "lad"],
                            f'{str(self)}__n_iter_no_change': [None, 5, 50, 100]}

        model = GradientBoostingRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "gbr"


class MLPModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {f'{str(self)}__hidden_layer_sizes': [(10, 4), (100,), (100, 50), (100, 150)],
                            f'{str(self)}__activation': ['logistic', 'relu'],
                            f'{str(self)}__solver': ["sgd", "adam"],
                            f'{str(self)}__learning_rate_init': [1e-2, 1e-3, 1e-4],
                            f'{str(self)}__learning_rate': ["constant", "adaptive"]}

        model = MLPRegressor(max_iter=10000)
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "mlp"


class XGBoostConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {
            f'{str(self)}__n_estimators': [400, 700, 1000],
            f'{str(self)}__colsample_bytree': [0.7, 0.8],
            f'{str(self)}__max_depth': [15, 20, 25],
            f'{str(self)}__reg_alpha': [1.1, 1.2, 1.3],
            f'{str(self)}__reg_lambda': [1.1, 1.2, 1.3],
            f'{str(self)}__subsample': [0.7, 0.8, 0.9]
        }
        model = XGBRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "XGBoost"