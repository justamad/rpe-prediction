from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVR, LinearSVR

import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("x.csv", sep=";").to_numpy()
y_labels = pd.read_csv("y.csv", sep=";")
y = y_labels['rpe'].to_numpy().reshape(-1, 1)
groups = y_labels['group'].to_numpy()

logo = LeaveOneGroupOut()

for counter, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index].reshape(-1), y[test_index].reshape(-1)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    svr_linear_opt = make_pipeline(StandardScaler(), LinearSVR(C=100, max_iter=1e4, verbose=10))
    svr_rbf = make_pipeline(StandardScaler(), SVR(kernel='linear', C=1, gamma=0.001, verbose=10))
    svr_linear = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1, gamma=0.001, verbose=10))
    # svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, gamma=0.001))

    # pipe = Pipeline([
    #     # ('constant', VarianceThreshold()),
    #     ('scaler', StandardScaler()),
    #     ('estimator', svr)
    # ])

    # pipe.fit(X_train, y_train)
    # print(pipe.score(X_train, y_train))

    svr_linear_opt.fit(X, y)
    print("linear opt: ", svr_linear_opt.score(X, y))

    svr_linear.fit(X, y)
    print("linear: ", svr_linear.score(X, y))

    svr_rbf.fit(X, y)
    print("rbf: ", svr_rbf.score(X, y))

    # pred_y = pipe.predict(X_test)

    # plt.plot(y_test, label="Ground Truth")
    # plt.plot(pred_y, label="Predictions")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{counter}_result.png")
    # plt.close()
    # plt.clf()
    # plt.cla()
