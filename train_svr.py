from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.processing import normalize_into_interval

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TKAgg")

sc_x = StandardScaler()
sc_y = StandardScaler()

X_train = pd.read_csv("X_train.csv", sep=";").to_numpy()
y_train = pd.read_csv("y_train.csv", sep=";").to_numpy().reshape(-1, 1)

X_test = pd.read_csv("X_test.csv", sep=";").to_numpy()
y_test = pd.read_csv("y_test.csv", sep=";").to_numpy().reshape(-1, 1)

X_train = sc_x.fit_transform(X_train)
y_train = normalize_into_interval(y_train, -1, 1)

X_test = sc_x.fit_transform(X_test)
y_test = normalize_into_interval(y_test, -1, 1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=2.0, epsilon=0.3))
svr = SVR(kernel='rbf', C=3.0, epsilon=0.01)
svr.fit(X_test, y_test)
print(svr.score(X_test, y_test))

pred_y = svr.predict(X_train)
plt.plot(y_train, label="Ground Truth")
plt.plot(pred_y, label="Predictions")
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig("output.png")
