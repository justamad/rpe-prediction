from os.path import join

import pandas as pd


def create_model_result_tables(df: pd.DataFrame, dst_path: str):
    metrics = {
        "r2": "$R^{2}$",
        "neg_mean_squared_error": "MSE",
        "neg_mean_absolute_error": "MAE",
        "mean_absolute_percentage_error": "MAPE",
    }

    values = []
    models = df["model"].unique()
    for model in models:
        sub_df = df[df["model"] == model].sort_values(by="mean_test_r2", ascending=False).iloc[0]

        row = {}
        for metric in metrics.keys():
            row[metric] = f"${sub_df[f'mean_test_{metric}']:0.2f} \\pm {sub_df[f'std_test_{metric}']:0.2f}$"

        values.append(row)

    final_df = pd.DataFrame(values, index=list(map(lambda model: model.upper(), models)))
    final_df = final_df.sort_values(by="r2", ascending=True).rename(columns=metrics).T
    final_df.to_latex(join(dst_path, "train_results.txt"), escape=False)
