from os.path import join
from scipy.stats import pearsonr, spearmanr

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def normalize_df(df):
    return (df - df.mean()) / df.std()


def read_kmeter_json_file(path, subject):
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
        set_df = set_df[(set_df["duration"] > 2) & (set_df["duration"] < 5)]
        set_df = set_df[set_df["rep_range"] > 30.0]

        set_df = set_df.mean(axis=0).to_frame().T  # Aggregate over reps

        set_df["set"] = set_counter
        set_df["rpe"] = rpe_values[set_counter]
        total_df = pd.concat([total_df, set_df], ignore_index=True)

    total_df["subject"] = subject
    total_df.iloc[:, :-3] = normalize_df(total_df.iloc[:, :-3])
    return total_df


def plot_data(df, subject):
    for param in ["peakSpeed", "powerAvg", "powerCon", "powerEcc", "rep_force", "rep_range"]:
        data = df[param].values
        # data = (data - data.mean()) / (data.max() - data.min())
        # data = (data - data.mean()) / data.std()
        la = spearmanr(data, df["rpe"].values)
        label = f"{param}: {la[0]:.2f}"
        plt.scatter(df["set"], data, label=label)
        # plt.plot(data, label=param)

    plt.scatter(df["set"], df["rpe"], label="rpe")
    plt.title(subject)
    plt.legend()
    plt.show()


path = "../../../../Volumes/INTENSO/RPE_Data/"
total_df = pd.DataFrame()
for subject in os.listdir(path):
    df = read_kmeter_json_file(path, subject)
    # plot_data(df, subject)
    total_df = pd.concat([total_df, df], ignore_index=True)

# df = read_kmeter_json_file(path, "857F1E")
# plot_data(total_df, "all")



total_df.to_csv("flywheel_mean.csv", index=False)

