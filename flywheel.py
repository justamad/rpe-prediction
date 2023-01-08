from os.path import join
from scipy.stats import pearsonr, spearmanr
from scipy.integrate import simps

import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


def normalize_df(df):
    return (df - df.mean()) / df.std()


def read_kmeter_json_file(path, subject, aggregate=False):
    rpe_file = join(path, subject, "rpe_ratings.json")
    with open(rpe_file) as f:
        rpe_values = json.load(f)
        rpe_values = np.array(rpe_values["rpe_ratings"])

    # rpe_values = (rpe_values - rpe_values.min()) / (rpe_values.max() - rpe_values.min())

    flywheel_file = join(path, subject, "kmeter.json")
    with open(flywheel_file) as f:
        content = json.load(f)

    total_df = pd.DataFrame()
    for set_counter, set_data in enumerate(content):
        print(f"subject: {subject}, set: {set_counter}, nr_reps: {len(set_data['training_rep'])}, t={set_data['training_rep'][0]['entry_time']}")

        if set_counter >= 12:
            continue

        set_df = pd.concat([pd.Series(rep) for rep in set_data["training_rep"]], axis=1, ignore_index=True).T
        set_df.drop(["entry_time", "id", "is_old_data", "set_id", "status"], axis=1, inplace=True)

        if not aggregate:
            set_df["nr_rep"] = np.arange(len(set_df))

        # set_df = set_df[(set_df["duration"] > 2) & (set_df["duration"] < 5)]
        # set_df = set_df[set_df["rep_range"] > 30.0]

        if aggregate:
            columns = list(set_df.columns)
            # set_df = set_df.mean(axis=0).to_frame().T  # Aggregate over reps
            set_df = set_df.aggregate(simps, axis=0).to_frame().T
            set_df.columns = columns

        set_df["nr_set"] = set_counter
        set_df["rpe"] = rpe_values[set_counter]
        total_df = pd.concat([total_df, set_df], ignore_index=True)

    total_df["subject"] = subject
    # if not aggregate:
    #     total_df.iloc[:, :-4] = normalize_df(total_df.iloc[:, :-4])
    # else:
    #     total_df.iloc[:, :-3] = normalize_df(total_df.iloc[:, :-3])
    return total_df


def plot_data(df, subject):
    params = ["peakSpeed", "powerAvg", "powerCon", "powerEcc", "rep_force", "rep_range", "duration"]
    fig, axs = plt.subplots(1, len(params), figsize=(20, 10))

    for c, param in enumerate(params):
        data = df[param].values
        roll_mean = pd.Series(data).rolling(window=10).mean()
        roll_std = pd.Series(data).rolling(window=10).std()

        spearman = spearmanr(data, df["rpe"].values)[0]
        label = f"{param}: SPC={spearman:.2f}"  # , R2={r2:.2f}"
        # plt.scatter(df["set"], data, label=label)
        axs[c].plot(data, label=label)
        axs[c].plot(roll_mean)
        axs[c].plot(roll_std)

    plt.scatter(df["nr_set"], df["rpe"], label="rpe")
    plt.title(subject)
    plt.legend()
    # plt.savefig(f"{subject}.png")
    plt.show()
    # plt.close()


def plot_interactive(df, subject):
    fig = px.scatter(
        data_frame=df,
        x=df.index,
        y="peakSpeed",
        color="nr_rep",
        # size="speed",
        hover_data=["duration", "peakSpeed", "powerAvg", "powerCon", "powerEcc", "rep_force", "rep_range", "nr_rep", "rpe"],
        title=f"{subject}",
        # opacity=0.2,
        width=1000,
        height=1000,
    )
    fig.show()


path = "../../../../Volumes/INTENSO/RPE_Data/"
total_df = pd.DataFrame()
for subject in os.listdir(path):
    df = read_kmeter_json_file(path, subject, aggregate=False)
    plot_data(df, subject)
    # plot_interactive(df, subject)
    total_df = pd.concat([total_df, df], ignore_index=True)

# total_df = read_kmeter_json_file(path, "857F1E")
# plot_data(total_df, "all")
# total_df.to_csv("flywheel.csv", index=False)
