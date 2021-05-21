from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR, LinearSVR

import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy().reshape(-1, 1)
groups = y_labels['groups'].to_numpy()

logo = LeaveOneGroupOut()
mm_scale = StandardScaler()

for counter, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    y_train = mm_scale.fit_transform(y_train).reshape(-1)
    y_test = mm_scale.fit_transform(y_test).reshape(-1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, gamma=0.001))
    # svr = make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5))
    svr.fit(X_train, y_train)
    print(svr.score(X_train, y_train))

    pred_y = svr.predict(X_test)

    plt.plot(y_test, label="Ground Truth")
    plt.plot(pred_y, label="Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{counter}_result.png")
    plt.close()
    plt.clf()
    plt.cla()
