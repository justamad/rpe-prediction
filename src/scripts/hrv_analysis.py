from string import ascii_uppercase

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import extract_dataset_input_output, normalize_data_by_subject
from src.plot import create_correlation_heatmap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.ml import eliminate_features_with_rfe
from sklearn.svm import SVR
from sklearn.feature_selection import RFE


def train_linear_regression(X: pd.DataFrame, y: pd.DataFrame):
    X = normalize_data_by_subject(X, y)
    X.fillna(0, inplace=True)

    y = y["rpe"]
    _, rfe_df = eliminate_features_with_rfe(X, y, 1, 1)
    print(rfe_df)

    # model = SVR(kernel='linear')
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    print("MSE: {}".format(mean_squared_error(y, y_pred)))

    print("Feature Coefficients:", model.coef_)
    print("length: ", len(model.coef_))

    coefficients = model.coef_
    # coefficients = model.support_

    # Display feature importance graphically
    plt.bar(range(len(coefficients)), coefficients)
    plt.xticks(range(len(coefficients)), X.columns, rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Coefficient Magnitude')
    plt.title('Feature Importance in Linear Regression')
    plt.tight_layout()
    plt.savefig("feature_importance.pdf")
    plt.close()

    plt.plot(y, label="true")
    plt.plot(y_pred, label="pred")
    plt.legend()
    plt.savefig("predictions.pdf")


def correlation_map(df: pd.DataFrame):
    values = {}
    for pseudonym, (subject_name, sub_df) in zip(ascii_uppercase, df.groupby("subject")):
        rpe = sub_df["rpe"]
        sub_df.drop(columns=["subject", "rpe", "set_id"], inplace=True)
        corr_df = sub_df.corrwith(rpe)
        values[pseudonym] = corr_df

    corr_df = pd.DataFrame(values).T
    mean_corr = corr_df.abs().mean(axis=0)
    columns = mean_corr.sort_values(ascending=False).index[:10]
    corr_df = corr_df[columns]

    corr_df.columns = [col.replace("HRV_", "").replace("s^2", "$s^2$").replace("%", "\%") for col in corr_df.columns]
    create_correlation_heatmap(corr_df, "hrv_correlation.pdf")


if __name__ == "__main__":
    df = pd.read_csv("data/training/full_stat.csv", index_col=0)

    drop_columns = []
    for prefix in ["FLYWHEEL", "PHYSILOG", "KINECT"]:
        drop_columns += [col for col in df.columns if prefix in col]

    df.drop(columns=drop_columns, inplace=True, errors="ignore")
    correlation_map(df)

    # X, y = extract_dataset_input_output(df=df, labels="rpe")
    # print(X.shape)
    # train_linear_regression(X, y)
