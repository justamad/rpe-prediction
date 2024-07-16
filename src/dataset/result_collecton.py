import numpy as np
import pandas as pd


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    result_df = pd.DataFrame()
    for model in df["model"].unique():
        model_df = df[df["model"] == model]

        for subject_name in model_df["subject"].unique():
            subject_df = model_df[model_df["subject"] == subject_name]

            data = {"ground_truth": [], "prediction": [], "set_id": []}

            for set_id in subject_df["set_id"].unique():
                set_df = subject_df[subject_df["set_id"] == set_id]
                data["ground_truth"].append(set_df["ground_truth"].to_numpy().mean())
                data["prediction"].append(np.average(set_df["prediction"]))

                data["set_id"].append(set_id)

            temp_df = pd.DataFrame(data)
            temp_df["model"] = model
            temp_df["subject"] = subject_name
            result_df = pd.concat([result_df, temp_df])

    return result_df
