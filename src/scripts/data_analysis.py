import numpy as np
import pandas as pd
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from os.path import join
from scipy.stats import spearmanr, linregress
from src.plot import plot_settings as ps
from string import ascii_uppercase
from argparse import ArgumentParser


UNITS = {
    "duration": "Duration [s]",
    "rep_range": "Range [cm]",
    "powerAvg": "Power (Avg) [Watt]",
    "powerCon": "Power (Con) [Watt]",
    "powerEcc": "Power (Ecc) [Watt]",
    "peakSpeed": "Velocity [cm/s]",
    "rep_force": "Force [N]",
}


def plot_correlation_per_subject(df: pd.DataFrame, feature: str, subject_name: str, src_path: str):
    df.reset_index(inplace=True, drop=True)
    feature_column = UNITS[feature]

    plt.figure(figsize=(ps.text_width * ps.cm, ps.text_width * 0.5 * ps.cm), dpi=ps.dpi)
    ax = plt.gca()
    ax.twinx()
    plt.xlabel("Repetitions")
    plt.ylabel("RPE")
    ax.set_ylabel("Watts")
    ax.set_xlabel("Repetitions")

    mean_df = sub_df.groupby("set_id").mean()
    mean_power = []
    for set_nr in sub_df["set_id"].unique():
        set_df = sub_df[sub_df["set_id"] == set_nr]
        x_min = set_df.index[0]
        x_max = set_df.index[-1]
        mean_power.extend([df[feature_column].loc[x_min:x_max].mean() for i in range(x_max - x_min + 1)])

        if set_nr % 2 == 0:
            plt.axvspan(x_min, x_max + 1, facecolor="0.7", alpha=0.1)

    lns1 = plt.plot(df["rpe"], color="0.0", label="RPE", linewidth=1)
    plt.ylim(10, 21)

    # slope, intercept, r, p, se = linregress(np.arange(len(mean_df)), mean_df[feature_column])
    # x_seq = np.linspace(0, len(sub_df), num=10)
    # lns2 = ax.plot(x_seq, intercept + slope * x_seq, color="0.2", lw=1.0, label="Regression Repetitions")
    lns2 = ax.plot(mean_power, color="0.6", label="Average Power", linewidth=1)

    # Create one legend for both y-axes
    lns = lns1 + lns2
    labs = [label.get_label() for label in lns]
    plt.legend(lns, labs, loc=0)

    spearman, _ = spearmanr(mean_df["rpe"], mean_df[feature_column])
    ax.scatter(df.index, df[feature_column], color="0.6", label=feature, s=2)

    plt.title(f"Spearmans $\\rho$={spearman:.2f}")
    plt.tight_layout()
    plt.savefig(join(src_path, f"{subject_name}.png"))
    plt.close()


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

    return pd.DataFrame(rpe_values).T


def plot_rpe_heatmap(rpe_df: pd.DataFrame, src_path: str):
    plt.figure(figsize=(ps.text_width * 0.5 * ps.cm, ps.text_width * 0.5 * ps.cm), dpi=ps.dpi)

    ticks = np.arange(11, 21)
    boundaries = np.arange(10.5, 21.5)
    cmap = matplotlib.cm.get_cmap("YlGnBu", len(ticks))
    sns.heatmap(data=rpe_df, cmap=cmap, annot=False, fmt='.2f', cbar_kws={'ticks': ticks, 'boundaries': boundaries})

    plt.ylabel("Subject")
    plt.xlabel("Set")
    plt.tight_layout()
    plt.savefig(join(src_path, "rpe_heatmap.pdf"))
    plt.close()


def plot_rpe_histogram(rpe_df: pd.DataFrame, src_path: str):
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
    plt.savefig(join(src_path, "rpe_histogram.pdf"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--src_path", type=str, dest="src_path", default="plots")
    args = parser.parse_args()

    os.makedirs(args.src_path, exist_ok=True)

    rpe_df = read_json_rpe_files("data/processed")
    plot_rpe_heatmap(rpe_df, args.src_path)
    plot_rpe_histogram(rpe_df, args.src_path)

    data_df = pd.read_csv("data/training/full_stat.csv", index_col=0)
    drop_columns = []
    for sensors in ["PHYSILOG", "HRV", "KINECT"]:
        drop_columns += [col for col in data_df.columns if sensors in col]

    data_df.drop(columns=drop_columns, inplace=True)
    data_df.rename(columns={col: col.replace("FLYWHEEL_", '') for col in data_df.columns}, inplace=True)
    data_df.rename(columns=UNITS, inplace=True)

    values = {}
    for pseudonym, (subject_name, sub_df) in zip(ascii_uppercase, data_df.groupby("subject")):
        sub_df.drop(columns=["subject"], inplace=True)
        plot_correlation_per_subject(sub_df, "powerAvg", subject_name, src_path=args.src_path)
        set_df = sub_df.groupby("set_id").mean()
        correlation_df = set_df.iloc[:, :7].corrwith(set_df["rpe"])
        values[pseudonym] = correlation_df

    correlation_df = pd.DataFrame(values).T
    mean_corr = correlation_df.mean(axis=0)  # .abs()
    print(mean_corr)
    create_correlation_heatmap(correlation_df)
