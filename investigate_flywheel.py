from src.dataset import normalize_data_by_subject
from scipy.stats import pearsonr

import pandas as pd
import plotly.express as px


df = pd.read_csv("flywheel.csv", index_col=False)

for subject in df["subject"].unique():
    sub_df = df[df["subject"] == subject]
    # df = normalize_data_by_subject(df)
    print(sub_df.rpe.value_counts())

    correlation, _ = pearsonr(sub_df.powerAvg, sub_df.nr_rep)

    fig = px.scatter(
        data_frame=sub_df,
        x="duration",
        y="powerAvg",
        color="rpe",
        # size="subject",
        hover_data=["duration", "peakSpeed", "powerAvg", "powerCon", "powerEcc", "rep_force", "rep_range", "nr_rep", "rpe",
                    "nr_set", "subject"],
        # title=f"{sub_df['rpe'].value_counts()}",
        title=f"{subject}, correlation RPE, powerAvg: {correlation}",
        # opacity=0.2,
        width=1000,
        height=1000,
    )
    fig.show()

