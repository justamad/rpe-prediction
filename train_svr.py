from src.processing import normalize_into_interval
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR

import pandas as pd
import matplotlib.pyplot as plt

sc_x = StandardScaler()
sc_y = StandardScaler()

X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy().reshape(-1, 1)
groups = y_labels['groups'].to_numpy()

logo = LeaveOneGroupOut()

for counter, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = sc_x.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)  # , -1, 1)

    X_test = sc_x.fit_transform(X_test)
    y_test = sc_y.fit_transform(y_test)  # , -1, 1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=2.0, epsilon=0.3))
    svr = SVR(kernel='rbf', C=1.0, gamma=0.001)
    svr.fit(X_test, y_test)
    print(svr.score(X_test, y_test))

    pred_y = svr.predict(X_train)

    plt.plot(y_train, label="Ground Truth")
    plt.plot(pred_y, label="Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{counter}_result.png")
    plt.close()
    plt.clf()
    plt.cla()
