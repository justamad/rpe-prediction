from src.processing import normalize_into_interval
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVR

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKAgg")

sc_x = StandardScaler()
sc_y = StandardScaler()

X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy()
groups = y_labels['groups'].to_numpy()

logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)
logo.get_n_splits(groups=groups)  # 'groups' is always required

for counter, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_train = sc_x.fit_transform(X_train)
    y_train = normalize_into_interval(y_train, -1, 1)

    X_test = sc_x.fit_transform(X_test)
    y_test = normalize_into_interval(y_test, -1, 1)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    # svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=2.0, epsilon=0.3))
    svr = SVR(kernel='rbf', C=2.0, epsilon=0.1)
    svr.fit(X_test, y_test)
    print(svr.score(X_test, y_test))

    pred_y = svr.predict(X_train)

    plt.plot(y_train, label="Ground Truth")
    plt.plot(pred_y, label="Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{counter}_result.png")
    # plt.show()
    plt.close()
    plt.clf()
    plt.cla()
