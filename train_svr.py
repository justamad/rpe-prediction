from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TKAgg")


X_train = pd.read_csv("X_train.csv", sep=";").to_numpy()
y_train = pd.read_csv("y_train.csv", sep=";").to_numpy().reshape(-1)

X_test = pd.read_csv("X_test.csv", sep=";").to_numpy()
y_test = pd.read_csv("y_test.csv", sep=";").to_numpy().reshape(-1)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
svr.fit(X_train, y_train)
print(svr.score(X_train, y_train))

pred_y = svr.predict(X_test)
plt.plot(y_test, label="Ground Truth")
plt.plot(pred_y, label="Predictions")
plt.legend()
plt.tight_layout()
plt.show()
