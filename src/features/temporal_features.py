import pandas as pd


def calculate_temporal_features(X: pd.DataFrame, y: pd.DataFrame, folds: int = 2) -> pd.DataFrame:
    total_df = pd.DataFrame()
    for subject in y["subject"].unique():
        mask = y["subject"] == subject
        sub_df = X.loc[mask]

        data_frames = [sub_df.diff(periods=period).add_prefix(f"GRAD_{period:02d}_") for period in range(1, folds + 1)]
        temp_df = pd.concat([sub_df] + data_frames, axis=1)
        temp_df.fillna(0, inplace=True)
        total_df = pd.concat([total_df, temp_df])

    total_df.reset_index(drop=True, inplace=True)
    return total_df
