from scipy.stats import pearsonr, linregress
from src.dataset import filter_outliers_z_scores
from os.path import join

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
import seaborn as sns
import plot_settings as ps
import numpy as np
import pandas as pd


df = pd.read_csv("flywheel.csv", index_col=False)

units = {
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
    plt.savefig(join("../plots", f"{subject_name}_{feature}.png"))
    return pear_set


def create_correlation_heatmap(df: pd.DataFrame):
    column_names = {
        "duration": "Duration",
        "peakSpeed": "Peak Speed",
        "powerAvg": "Average Power",
        "powerCon": "Concentric Power",
        "powerEcc": "Eccentric Power",
        "rep_force": "Repetition Force",
        "rep_range": "Repetition Range",
    }
    df.rename(columns=column_names, inplace=True)

    plt.figure(figsize=(ps.image_width * 2 * ps.cm, ps.image_width * 2 * ps.cm), dpi=300)
    ax = sns.heatmap(df, fmt=".2f", vmin=-1, vmax=1, linewidth=0.5, annot=True)  # , cmap=colormap)
    plt.ylabel("Subject")
    plt.xlabel("Feature")
    plt.tight_layout()
    # plt.show()
    plt.savefig("correlations.pdf", dpi=300)
    plt.close()


correlation_df = pd.DataFrame()

for subject in df["subject"].unique():
    sub_df = df[df["subject"] == subject]
    l1 = len(sub_df)
    sub_df = filter_outliers_z_scores(sub_df, "duration", sigma=2.0)
    sub_df = filter_outliers_z_scores(sub_df, "rep_range", sigma=2.0)
    sub_df = filter_outliers_z_scores(sub_df, "powerCon", sigma=3.0)
    sub_df = filter_outliers_z_scores(sub_df, "powerEcc", sigma=3.0)
    sub_df = filter_outliers_z_scores(sub_df, "powerAvg", sigma=3.0)
    l2 = len(sub_df)
    print(f"Filtered {l1 - l2} outliers for subject {subject}")
    sub_df.reset_index(inplace=True)

    features = ["duration", "peakSpeed", "powerAvg", "powerCon", "powerEcc", "rep_force", "rep_range"]

    correlation_values = []
    for feature in features:
        plt.close()
        plt.figure(figsize=(ps.image_width * 2 * ps.cm, ps.image_width * ps.cm), dpi=300)
        corr_value = plot_correlation_per_subject(sub_df, feature)
        correlation_values.append(corr_value)

    corr_df = pd.DataFrame([[subject] + correlation_values], columns=["subject"] + features)
    correlation_df = pd.concat([correlation_df, corr_df], axis=0)

correlation_df.set_index("subject", inplace=True)
correlation_df.to_csv("correlation.csv", index=True)

create_correlation_heatmap(correlation_df)
