import numpy as np
import pandas as pd


def split_data_to_pseudonyms(x: pd.DataFrame, y: pd.DataFrame, train_p: float = 0.8, random_seed: int = None):
    subject_names = sorted(y['name'].unique())
    nr_subjects = int(len(subject_names) * train_p)

    if random_seed is not None:
        np.random.seed(random_seed)

    train_subjects = np.random.choice(subject_names, nr_subjects, replace=False)
    train_idx = y['name'].isin(train_subjects)

    return x.loc[train_idx].copy(), y.loc[train_idx].copy(), x.loc[~train_idx].copy(), y.loc[~train_idx].copy()


def normalize_rpe_values_min_max(df: pd.DataFrame):
    subjects = df['group'].unique()
    for subject_id in subjects:
        mask = df['group'] == subject_id
        df_subject = df[mask]
        min_rpe, max_rpe = df_subject['rpe'].min(), df_subject['rpe'].max()
        df.loc[mask, 'rpe'] = (df_subject['rpe'] - min_rpe) / (max_rpe - min_rpe)

    return df
