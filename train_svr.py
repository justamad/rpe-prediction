from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pandas as pd
import matplotlib
matplotlib.use("TKAgg")

import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv("X.csv", sep=";").to_numpy()
y = pd.read_csv("y.csv", sep=";").to_numpy().reshape(-1)

print(X.shape, y.shape)
svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
svr.fit(X, y)
print(svr.score(X, y))

pred_y = svr.predict(X)
plt.plot(y, label="Ground Truth")
plt.plot(pred_y, label="Predictions")
plt.legend()
plt.tight_layout()
plt.show()
