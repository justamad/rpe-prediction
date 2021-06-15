from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


class LearningModelBase(object):

    def __init__(self, model, parameters):
        self.scaler = StandardScaler()
        self.model = model
        self.parameters = parameters

    def get_trial_data_dict(self):
        """
        Returns the collection of parameters for grid search
        @return: dictionary that holds collection of parameters for grid search
        """
        return {
            'model': self.model,
            'scaler': self.scaler,
            'parameters': self.parameters,
            'learner_name': str(self),
            'balancer': RandomOverSampler(),
        }


class SVRModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {'svr__kernel': ('linear', 'rbf'),
                            'svr__gamma': [1e-3, 1e-4],
                            'svr__C': [1e0, 1e1, 1e2, 1e3], }

        model = SVR()
        super().__init__(model=model, parameters=tuned_parameters)

    def __repr__(self):
        return "svr"


class KNNModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {'knn__n_neighbors': (5, 10, 15),
                            'knn_weights': ['uniform', 'distance'],
                            'knn_algorithm': ['ball_tree', 'kd_tree']}

        model = KNeighborsRegressor()
        super().__init__(model=model, parameters=tuned_parameters)

    def __repr__(self):
        return "knn"


class RFModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {'rf_n_estimators': (50, 100, 150, 200),
                            'rf_criterion': ('mse', 'mae')}

        model = RandomForestRegressor()
        super().__init__(model=model, parameters=tuned_parameters)

    def __repr__(self):
        return "rf"
