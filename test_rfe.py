from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.model_selection import LeaveOneGroupOut

import pandas as pd

X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy().reshape(-1, 1)
groups = y_labels['group'].to_numpy()

logo = LeaveOneGroupOut()

estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, n_jobs=-1)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
