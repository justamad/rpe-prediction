from os.path import join
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats

import pandas as pd
import numpy as np


def create_train_table(df: pd.DataFrame, dst_path: str):
    metrics = {
        "r2": "$R^{2}$",
        "mean_absolute_percentage_error": "MAPE",
        "neg_mean_squared_error": "MSE",
        "neg_mean_absolute_error": "MAE",
    }

    data_entries = []
    models = sorted(df["model"].unique())
    for model in models:
        sub_df = df[df["model"] == model].sort_values(by="mean_test_r2", ascending=False).iloc[0]

        row = {}
        for metric in metrics.keys():
            row[metric] = f"${sub_df[f'mean_test_{metric}']:0.2f} \\pm {sub_df[f'std_test_{metric}']:0.2f}$"

        data_entries.append(row)

    final_df = pd.DataFrame(data_entries, index=list(map(lambda m: m.upper(), models)))
    final_df = final_df.rename(columns=metrics).T
    final_df.to_latex(join(dst_path, "train_results.txt"), escape=False)


def create_retrain_table(results: pd.DataFrame, dst_path: str) -> pd.DataFrame:
    metrics = {
        "MSE": lambda x, y: mean_squared_error(x, y, squared=True),
        "RMSE": lambda x, y: mean_squared_error(x, y, squared=False),
        "MAE": mean_absolute_error,
        "MAPE": lambda x, y: mean_absolute_percentage_error(x, y) * 100,
        "$R^{2}$": r2_score,
        "Spearman": lambda x, y: stats.spearmanr(x, y)[0]
    }

    data_entries = []
    models = sorted(results["model"].unique())
    for model in models:
        model_df = results[results["model"] == model]

        test_subjects = {key: [] for key in metrics.keys()}
        for subject in model_df["subject"].unique():
            subject_df = model_df[model_df["subject"] == subject]

            for metric, func in metrics.items():
                test_subjects[metric].append(func(subject_df["ground_truth"], subject_df["prediction"]))

        res = {metric: f"${np.mean(values):.2f} \\pm {np.std(values):.2f}$" for metric, values in test_subjects.items()}
        data_entries.append(res)

    final_df = pd.DataFrame.from_records(data_entries, index=list(map(lambda x: x.upper(), models))).T
    final_df.to_latex(join(dst_path, "retrain_results_latex.txt"), escape=False)
    return final_df
