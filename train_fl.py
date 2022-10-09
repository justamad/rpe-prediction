from src.dataset import extract_dataset_input_output, normalize_subject_rpe, normalize_data_by_subject, discretize_subject_rpe
from src.plot import plot_prediction_results_for_sets
from sklearn.svm import SVR
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("flywheel.csv", index_col=False)
df = normalize_data_by_subject(df)
# df = discretize_subject_rpe(df)

X, y = extract_dataset_input_output(df)
# y = normalize_subject_rpe(y)

# X = X[["nr_set"]]

for subject in df["subject"].unique():

    mask = y["subject"] == subject

    model = SVR(kernel="rbf", C=1e2, gamma=0.001)
    X_train = X.loc[~mask, :]
    y_train = y.loc[~mask, :]

    X_test = X.loc[mask, :]
    y_test = y.loc[mask, :]

    X_train, y_train = SMOTE().fit_resample(X_train, y_train["rpe"])

    # model.fit(X_train, y_train["rpe"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test["prediction"] = y_pred
    plot_prediction_results_for_sets(y_test)

    mse = mean_squared_error(y_test["rpe"], y_pred)
    mae = mean_absolute_error(y_test["rpe"], y_pred)
    mape = mean_absolute_percentage_error(y_test["rpe"], y_pred)

    plt.plot(np.arange(len(y_pred)), y_pred, label="predicted")
    plt.plot(np.arange(len(y_pred)), y_test["rpe"], label="actual")
    plt.title(f"{subject}, MSE: {mse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}")
    plt.legend()
    plt.show()
    plt.close()
