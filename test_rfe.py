from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, GenericUnivariateSelect, VarianceThreshold
from sklearn.datasets import make_regression

import pandas as pd
import numpy as np

data = pd.read_csv("x.csv", sep=";")
print(data.columns)

X = data.to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy()
# groups = y_labels['group'].to_numpy()
# logo = LeaveOneGroupOut()

# max = -1
# nr_features = -1
# X = X[:max, :nr_features]
# y = y[:max]
print(X.shape, y.shape)
# X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)


estimator = LinearSVR(C=100, max_iter=1e4)
# estimator = SVR(kernel='rbf', C=1.0, gamma=0.001)
selector = RFE(estimator, n_features_to_select=30, step=20, verbose=10)
pipe = Pipeline([
    # ('constant', VarianceThreshold()),
    ('scaler', StandardScaler()),
    ('selector', selector),
    # ('estimator', estimator)
])

# selector = selector.fit(X, y)

pipe.fit(X, y)
print(pipe)
print(pipe.score(data.to_numpy(), y))
# print(selector.transform(X, y))
# selector = RFECV(estimator, step=1, n_jobs=-1)
# selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
# print(data.columns[selector.ranking_ == 1])
