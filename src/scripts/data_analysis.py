import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from os.path import join
from scipy.stats import pearsonr, linregress
from src.plot import plot_settings as ps
from string import ascii_uppercase


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
    plt.savefig(join("../plots_wichtig", f"{subject_name}_{feature}.png"))
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

    plt.figure(figsize=(ps.column_width * 1.2 * ps.cm, ps.column_width * ps.cm), dpi=ps.dpi)
    cbar_kws = {"label": "Pearson's Correlation Coefficient", }  # "ticks_position": "right"
    ax = sns.heatmap(df, fmt=".2f", vmin=-1, vmax=1, linewidth=0.5, annot=True, cbar_kws=cbar_kws)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Subject")
    # plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig("correlations.pdf")
    plt.close()


if __name__ == "__main__":
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
        correlation_df = set_df.iloc[:, :7].corrwith(set_df['rpe'])
        values[pseudonym] = correlation_df

    correlation_df = pd.DataFrame(values).T
    mean_corr = correlation_df.mean(axis=0) # .abs()
    print(mean_corr)
    create_correlation_heatmap(correlation_df)
