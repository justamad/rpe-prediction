from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, GenericUnivariateSelect, VarianceThreshold
from sklearn.datasets import make_regression

import pandas as pd

X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy()
# groups = y_labels['group'].to_numpy()
# logo = LeaveOneGroupOut()

X = X[:1000, :10]
y = y[:1000]
print(X.shape, y.shape)
# X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)


scaler = StandardScaler()
estimator = SVR(kernel="linear", C=100)
selector = RFE(estimator, n_features_to_select=5, step=1, verbose=10)
pipe = Pipeline([
    # ('constant', VarianceThreshold()),
    ('scaler', scaler),
    # ('selector', selector),
    ('estimator', estimator)
])

# selector = selector.fit(X, y)

pipe.fit(X, y)
print(pipe.score(X, y))
# print(selector.transform(X, y))
# selector = RFECV(estimator, step=1, n_jobs=-1)
# selector = selector.fit(X, y)
# print(selector.support_)
# print(selector.ranking_)
