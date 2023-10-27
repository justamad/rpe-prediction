import json
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from os.path import join
from scipy.stats import pearsonr, linregress
from src.plot import plot_settings as ps
from string import ascii_uppercase


UNITS = {
    "duration": "Duration [s]",
    "rep_range": "Distance [cm]",
    "powerAvg": "Power [Watt]",
    "powerCon": "Power [Watt]",
    "powerEcc": "Power [Watt]",
    "peakSpeed": "Velocity [cm/s]",
    "rep_force": "Newton [N]",
}


def plot_correlation_per_subject(sub_df: pd.DataFrame, feature: str):
    subject_name = sub_df["subject"].unique()[0]
    ax = plt.gca()
    ax.twinx()

    plt.xlabel("Repetitions")
    plt.ylabel("RPE")
    ax.set_ylabel(units[feature])
    ax.set_xlabel("Repetitions")

    lns1 = plt.plot(sub_df["rpe"], color="red", label="RPE", linewidth=1)
    ax.scatter(sub_df.index, sub_df[feature], color="black", label=feature, s=2)

    plt.ylim(10, 21)

    pear, _ = pearsonr(sub_df["rpe"], sub_df[feature])

    # Plot regression line entire data frame
    slope, intercept, r, p, se = linregress(np.arange(len(sub_df)), sub_df[feature])
    x_seq = np.linspace(0, len(sub_df), num=10)
    lns2 = ax.plot(x_seq, intercept + slope * x_seq, color="green", lw=1.0, label="Regression Repetitions")

    # Create one legend for both y-axes
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=0)

    mean_set = []
    rpe_set = []
    for set_nr in sub_df["nr_set"].unique():
        set_df = sub_df[sub_df["nr_set"] == set_nr]
        x_min = set_df.index[0]
        x_max = set_df.index[-1]

        if set_nr % 2 == 0:
            plt.axvspan(x_min, x_max + 1, facecolor="gray", alpha=0.1)

        mean = set_df[feature].mean()
        mean_set.append(mean)
        rpe_set.append(set_df["rpe"].mean())

        ax.scatter((x_min + x_max) / 2, mean, color="red", s=10, marker="x")

        # ax.plot((xmin, xmax), (mean, mean), color="black", linewidth=1)

        # Plot regression line
        slope, intercept, r, p, se = linregress(np.arange(x_min, x_max + 1), set_df[feature])
        x_seq = np.linspace(x_min, x_max, num=50)
        ax.plot(x_seq, intercept + slope * x_seq, color="blue", lw=1.0)

    pear_set, _ = pearsonr(rpe_set, mean_set)
    slope, intercept, r, p, se = linregress(np.arange(len(mean_set)), mean_set)
    plt.title(f"Repetitions PCC={pear:.2f}, Sets PCC={pear_set:.2f}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(join("../plots_wichtig", f"{subject_name}_{feature}.png"))
    return pear_set


def create_correlation_heatmap(df: pd.DataFrame):
    plt.figure(figsize=(ps.text_width * ps.cm, ps.text_width * 0.8 * ps.cm), dpi=ps.dpi)
    cbar_kws = {"label": "Pearson's Correlation Coefficient", }  # "ticks_position": "right"
    sns.heatmap(df, fmt=".2f", vmin=-1, vmax=1, linewidth=0.5, annot=True, cbar_kws=cbar_kws)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig("correlations.pdf")
    plt.close()


def read_json_rpe_files(src_path: str) -> pd.DataFrame:
    rpe_values = {}
    for pseudonym, subject_folder in zip(ascii_uppercase, os.listdir(src_path)):
        json_content = json.load(open(join(src_path, subject_folder, "rpe_ratings.json")))
        rpe = list(map(int, json_content["rpe_ratings"]))
        rpe_values[pseudonym] = rpe + [np.nan for _ in range(12 - len(rpe))]

    return pd.DataFrame(rpe_values).T  #.astype(int)


def plot_rpe_heatmap(rpe_df: pd.DataFrame):
    plt.figure(figsize=(ps.text_width * 0.5 * ps.cm, ps.text_width * 0.5 * ps.cm), dpi=ps.dpi)

    ticks = np.arange(11, 21)
    boundaries = np.arange(10.5, 21.5)
    cmap = matplotlib.cm.get_cmap("YlGnBu", len(ticks))
    sns.heatmap(data=rpe_df, cmap=cmap, annot=False, fmt='.2f', cbar_kws={'ticks': ticks, 'boundaries': boundaries})

    plt.ylabel("Subject")
    plt.xlabel("Set")
    plt.tight_layout()
    plt.savefig("rpe_heatmap.pdf")
    plt.close()


def plot_rpe_histogram(rpe_df: pd.DataFrame):
    all_values = rpe_df.values.flatten()
    data = all_values[~np.isnan(all_values)].astype(int)

    plt.figure(figsize=(ps.text_width * 0.5 * ps.cm, ps.text_width * 0.5 * ps.cm), dpi=ps.dpi)
    hist, bins = np.histogram(data, bins=np.arange(np.min(data), np.max(data) + 2))
    plt.bar(bins[:-1], hist, width=1, align='center')  # Use 'center' alignment
    x_ticks = bins[:-1]  # Adjust the tick positions to the center of the bars
    x_labels = bins[:-1]
    plt.xticks(x_ticks, x_labels)
    plt.ylim(0, 40)

    for i, count in enumerate(hist):
        plt.text(x_ticks[i], count + 1, str(count), ha='center', va='bottom')

    plt.xlabel("RPE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("rpe_histogram.pdf")


if __name__ == "__main__":
    rpe_df = read_json_rpe_files("data/processed")
    plot_rpe_heatmap(rpe_df)
    plot_rpe_histogram(rpe_df)

    data_df = pd.read_csv("data/training/full_stat.csv", index_col=0)
    drop_columns = []
    for sensors in ["PHYSILOG", "HRV", "KINECT"]:
        drop_columns += [col for col in data_df.columns if sensors in col]

    data_df.drop(columns=drop_columns, inplace=True)
    data_df.rename(columns={col: col.replace("FLYWHEEL_", '') for col in data_df.columns}, inplace=True)

    values = {}
    for pseudonym, (subject_name, sub_df) in zip(ascii_uppercase, data_df.groupby("subject")):
        sub_df.drop(columns=["subject"], inplace=True)
        set_df = sub_df.groupby("set_id").mean()
        correlation_df = set_df.iloc[:, :7].corrwith(set_df["rpe"])
        values[pseudonym] = correlation_df

    correlation_df = pd.DataFrame(values).T
    correlation_df.rename(columns=UNITS, inplace=True)
    mean_corr = correlation_df.mean(axis=0)  # .abs()
    print(mean_corr)
    create_correlation_heatmap(correlation_df)
