import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from string import ascii_uppercase
from src.dataset import extract_dataset_input_output, normalize_data_by_subject, get_highest_correlation_features
from src.plot import create_correlation_heatmap, plot_sample_predictions
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser
from os.path import join


def train_linear_regression(X: pd.DataFrame, y: pd.DataFrame):
    X = normalize_data_by_subject(X, y)
    X.fillna(0, inplace=True)
    X = get_highest_correlation_features(X, y["rpe"], 10)

    result_df = pd.DataFrame()
    for subject in y["subject"].unique():
        mask = y["subject"] == subject
        X_train, y_train, X_test, y_test = X.loc[~mask], y.loc[~mask], X.loc[mask], y.loc[mask]
        model = LinearRegression()
        model.fit(X_train, y_train["rpe"])
        y_pred = model.predict(X_test)
        print("MSE: {}".format(mean_squared_error(y_test["rpe"], y_pred)))
        y_test["prediction"] = y_pred
        result_df = pd.concat([result_df, y_test], axis=0)

        # Visualize individual features along with predictions
        X_test = X_test.values
        y_test = y_test["rpe"]
        for feature_index in range(X.shape[1]):
            plt.figure(figsize=(8, 6))
            # Scatter plot of the original feature values
            plt.scatter(X_test[:, feature_index], y_test, label='Actual', color='blue', alpha=0.7)

            # Line plot of the predicted values
            sorted_indices = np.argsort(X_test[:, feature_index])
            plt.plot(X_test[:, feature_index][sorted_indices], y_pred[sorted_indices], label='Predicted', color='red',
                     linewidth=2)

            plt.title(f'Feature {feature_index + 1} vs. Predictions')
            plt.xlabel(f'Feature {feature_index + 1}')
            plt.ylabel('Target Variable (y)')
            plt.legend()
            plt.show()

    plot_sample_predictions(result_df, exp_name="rpe", dst_path=".", label_col="rpe")


def create_hrv_correlation_map(df: pd.DataFrame, file_name: str):
    values = {}
    for pseudonym, (subject_name, sub_df) in zip(ascii_uppercase, df.groupby("subject")):
        sub_df.drop(columns=["subject"], inplace=True)
        sub_df = sub_df.groupby("set_id").mean()
        rpe = sub_df["rpe"]
        # for feature in sub_df.columns:
        #     print(scipy.stats.pearsonr(sub_df[feature], rpe))
        corr_df = sub_df.corrwith(rpe)
        values[pseudonym] = corr_df

    corr_df = pd.DataFrame(values).T
    corr_df.drop(["rpe"], inplace=True, axis=1)
    mean_corr = corr_df.abs().mean(axis=0)
    columns = mean_corr.sort_values(ascending=False).index[:10]
    corr_df = corr_df[columns]

    corr_df.columns = [col.replace("HRV_", "").replace("s^2", "$s^2$").replace("%", "\%") for col in corr_df.columns]
    create_correlation_heatmap(corr_df, file_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="plots")
    args = parser.parse_args()

    df = pd.read_csv("data/training/full_stat.csv", index_col=0)
    drop_columns = []
    for prefix in ["FLYWHEEL", "PHYSILOG", "KINECT"]:
        drop_columns += [col for col in df.columns if prefix in col]
    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    create_hrv_correlation_map(df, join(args.src_path, "hrv_correlations.pdf"))
    X, y = extract_dataset_input_output(df=df, labels="rpe")
