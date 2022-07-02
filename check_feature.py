from src.plot import plot_feature_distribution_as_pdf
from src.utils import filter_outliers_z_scores, normalize_data_by_subject

import pandas as pd


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


df = pd.read_csv("X.csv", sep=";", index_col=0)
df = impute_dataframe(df)

mask = filter_outliers_z_scores(df)
df = df[mask]
df = df.dropna(axis=1, how="all")
df = normalize_data_by_subject(df)


plot_feature_distribution_as_pdf(df, "test.pdf")
print(df)
