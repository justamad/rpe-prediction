from src.plot import plot_feature_distribution_as_pdf
from src.utils import filter_outliers_z_scores

import pandas as pd


def impute_dataframe(df: pd.DataFrame):
    df = df.interpolate(method="linear", limit_direction="forward", axis=0)
    df = df.fillna(0)
    return df


df = pd.read_csv("X.csv", sep=";", index_col=0)
df = impute_dataframe(df)
df_y = pd.read_csv("y.csv", sep=";", index_col=0)

mask = filter_outliers_z_scores(df)
df["set"] = df_y["set"]
df["name"] = df_y["name"]
df = df[mask]
df = df.dropna(axis=1, how='all')

total_df = pd.DataFrame()
for name, group in df.groupby("name"):
    sub_df = group.iloc[:, :-2]
    group.iloc[:, :-2] = (sub_df - sub_df.min()) / (sub_df.max() - sub_df.min())
    total_df = pd.concat([total_df, group], ignore_index=True)

total_df = impute_dataframe(total_df)


plot_feature_distribution_as_pdf(total_df, "test.pdf")
print(df)
