from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, GenericUnivariateSelect, VarianceThreshold
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
            'selector': RFE(SVR('linear')),
            'constant_remover': VarianceThreshold()
        }


class SVRModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = [{'regression__kernel': ['linear'],  # 'rbf'],
                             'regression__gamma': [1e-3],  # 1e-4],
                             'regression__C': [1e0]}]  #, 1e1]}]  # , 1e2, 1e3]}]

        model = SVR()
        super().__init__(model=model, parameters=tuned_parameters)

    def __repr__(self):
        return "svr"
