from .plot_settings import column_width, cm, dpi, line_width, blob_size
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr, linregress
from os.path import join, exists
from os import makedirs
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoLocator

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


y_labels = {"rpe": "RPE [Borg Scale]", "hr": "Heart Rate [1/min]", "poweravg": "Power [Watts]",}
y_limits = {"rpe": (11, 21), "hr": (90, 200), "poweravg": (0, 300),}


def plot_sample_predictions(
        value_df: pd.DataFrame,
        exp_name: str,
        dst_path: str,
        pred_col: str = "prediction",
        label_col: str = "ground_truth"
):
    if exp_name not in y_labels or exp_name not in y_limits:
        raise ValueError(f"Unknown experiment '{exp_name}' for plotting predictions.")

    for subject_name in value_df["subject"].unique():
        subject_df = value_df[value_df["subject"] == subject_name]
        ground_truth = subject_df[label_col].values
        predictions = subject_df[pred_col].values

        r2 = r2_score(ground_truth, predictions)
        mape = mean_absolute_percentage_error(predictions, ground_truth)
        sp, _ = spearmanr(ground_truth, predictions)

        plt.figure(figsize=(0.8 * column_width * cm, 0.8 * column_width * cm), dpi=dpi)
        plt.plot(ground_truth, label="Ground Truth", c="lightgray")
        plt.plot(predictions, label="Prediction", c="black")
        plt.ylim(y_limits[exp_name])
        plt.title(f"$R^2$={r2:.2f}, MAPE={mape:.2f}, S={sp:.2f}")
        plt.xlabel("Repetition")
        plt.ylabel(y_labels[exp_name])
        plt.legend()
        plt.tight_layout()

        if not exists(dst_path):
            makedirs(dst_path)

        plt.savefig(join(dst_path, f"{subject_name}.pdf"))
        plt.clf()
        plt.close()


def evaluate_nr_features(df: pd.DataFrame, dst_path: str):
    plt.figure(figsize=(column_width * cm, column_width * cm * 0.65), dpi=dpi)

    nr_features = sorted(df["n_features"].unique())
    for model in df["model"].unique():
        sub_df = df[df["model"] == model]

        x_axis = []
        y_axis = []
        errors = []
        for nr_feature in nr_features:
            sub_sub_df = sub_df[sub_df["n_features"] == nr_feature].sort_values(by="mean_test_r2", ascending=False).iloc[0]
            x_axis.append(nr_feature)
            y_axis.append(sub_sub_df["mean_test_r2"])
            errors.append(sub_sub_df["std_test_r2"])

        # plt.errorbar(x_axis, y_axis, yerr=errors, label=model.upper())
        plt.plot(x_axis, y_axis, label=model.upper())

    # plt.ylim(0, 1)
    plt.xticks(nr_features)
    plt.legend()
    plt.xlabel("Number of Features")
    plt.ylabel("$R^2$")
    plt.tight_layout()

    if not exists(dst_path):
        makedirs(dst_path)

    plt.savefig(join(dst_path, "nr_features.pdf"))
    plt.clf()
    plt.close()


def plot_subject_correlations(df: pd.DataFrame, dst_path: str):
    metrics = []
    subjects = df["subject"].unique()
    for subject in df["subject"].unique():
        sub_df = df[df["subject"] == subject]
        metric, p_value = spearmanr(sub_df["ground_truth"], sub_df["prediction"])
        metrics.append(metric)
        print(f"{subject}: {metric:.2f} ({p_value:.2f})")

    fig, axs = plt.subplots(1, 1, figsize=(column_width * cm, column_width * cm / 2), dpi=dpi)
    axs.bar([f"{i+1:2d}" for i in range(len(subjects))], metrics, color="darkgray")
    # plt.title("Spearman's Rho")
    plt.ylim([0, 1])
    plt.xlabel("Subjects")
    plt.ylabel("Spearman's Rho")
    plt.tight_layout()

    if not exists(dst_path):
        makedirs(dst_path)

    plt.savefig(join(dst_path, "subject_performance.pdf"))
    # plt.show()
    plt.clf()
    plt.close()


def create_scatter_plot(
        df: pd.DataFrame,
        log_path: str,
        file_name: str,
        exp_name: str,
):
    max_limits = {"poweravg": 800}

    ground_truth = df.loc[:, "ground_truth"]
    prediction = df.loc[:, "prediction"]

    slope, intercept, r_value, p_value_1, std_error_1 = linregress(ground_truth, prediction)
    rmse = np.sqrt(mean_squared_error(ground_truth, prediction))
    r2 = r_value ** 2

    min_value = 0
    if exp_name in max_limits:
        max_value = max_limits[exp_name]
    else:
        max_value = max(ground_truth.max(), prediction.max())

    x_values = np.arange(int(min_value * 100), int(max_value * 100 + 50)) / 100
    y_values = intercept + slope * x_values

    fig = plt.figure(figsize=(0.75 * column_width * cm, 0.75 * column_width * cm), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(line_width)
    ax.spines["left"].set_linewidth(line_width)

    plt.plot(x_values, x_values, '-', color="gray", linewidth=line_width)
    plt.plot(x_values, y_values, '--', color="red", linewidth=line_width)

    sign = "+" if intercept > 0 else ""
    margin = (max_value - min_value) * 0.95
    plt.text(
        min_value + margin,
        max_value - margin,
        f"$RMSE={rmse:.2f}$\n$R^{{2}}={r2:.2f}$\n$y={slope:.2f}x{sign}{intercept:.2f}$",
        style="italic",
        horizontalalignment="right",
    )

    plt.scatter(ground_truth, prediction, s=blob_size, c="gray", alpha=0.5),
    plt.xlim(([min_value, max_value]))
    plt.ylim(([min_value, max_value]))

    locator = AutoLocator()
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)

    ax.set_xlabel(f"Ground Truth")
    ax.set_ylabel(f"Prediction")
    plt.tight_layout()

    if not exists(log_path):
        makedirs(log_path)

    plt.savefig(join(log_path, f"{file_name}_scatter.pdf"))
    plt.clf()
    plt.cla()
    plt.close()


def create_bland_altman_plot(
        df: pd.DataFrame,
        log_path: str,
        file_name: str,
        sd_limit: float = 1.96,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
):
    m1 = df.loc[:, "prediction"]
    m2 = df.loc[:, "ground_truth"]

    fig = plt.figure(figsize=(0.75 * column_width * cm, 0.75 * column_width * cm), dpi=dpi)
    ax = fig.add_subplot(111)

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    # Plot individual gait speed colors
    ax.scatter(means, diffs, s=blob_size, c="gray", alpha=0.5)
    ax.axhline(mean_diff, **{"color": "gray", "linewidth": 1, "linestyle": "--"})

    if x_min is not None and x_max is not None:
        plt.xlim(([x_min, x_max]))

    if y_min is not None and y_max is not None:
        plt.ylim(([y_min, y_max]))

    # Annotate mean line with mean difference.
    ax.annotate(
        f"Mean Diff:\n{mean_diff:.2f}",
        xy=(0.99, 0.5),
        horizontalalignment="right",
        verticalalignment="center",
        xycoords="axes fraction"
    )

    if sd_limit > 0:
        # half_ylim = (1.5 * sd_limit) * std_diff
        # ax.set_ylim(
        #     mean_diff - half_ylim,
        #     mean_diff + half_ylim
        # )
        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **{"color": "gray", "linewidth": 1, "linestyle": ":"})

        ax.annotate(f'-{sd_limit:.2f} SD: {lower:.2f}',
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    xycoords='axes fraction')
        ax.annotate(f'+{sd_limit:.2f} SD: {upper:.2f}',
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(
            mean_diff - half_ylim,
            mean_diff + half_ylim
        )

    ax.set_ylabel("Difference between two measurements")
    ax.set_xlabel("Average of two measurements")
    # plt.xlim(([min_value, max_value]))
    # plt.ylim(([min_value, max_value]))

    fig.tight_layout()

    if not exists(log_path):
        makedirs(log_path)

    plt.savefig(join(log_path, f"{file_name}_ba.pdf"))
    plt.clf()
    plt.cla()
    plt.close()
