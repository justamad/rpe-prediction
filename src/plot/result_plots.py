import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .plot_settings import text_width, cm, dpi, line_width, blob_size
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr, linregress
from os.path import join, exists
from os import makedirs
from matplotlib.ticker import MaxNLocator, AutoLocator

y_labels = {"rpe": "RPE [Borg Scale]", "poweravg": "Power [Watts]", }
y_limits = {"rpe": (10, 21), "poweravg": (0, 300), }
max_limits = {"poweravg": 800}  # Used to equally scale physical model and ML models

primary_color = "#d62728"
secondary_color = "#1f77b4"


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

        plt.figure(figsize=(text_width * cm * 0.5, text_width * cm * 0.5), dpi=dpi)
        plt.plot(ground_truth, label="Ground Truth", color=primary_color)
        plt.plot(predictions, label="Prediction", color=secondary_color)
        plt.ylim(y_limits[exp_name])
        plt.title(f"$R^2$={r2:.2f}, MAPE={mape:.2f}, $\\rho$={sp:.2f}")
        plt.xlabel("Repetition")
        plt.ylabel(y_labels[exp_name])
        plt.legend()
        plt.tight_layout()

        if not exists(dst_path):
            makedirs(dst_path)

        plt.savefig(join(dst_path, f"{subject_name}.pdf"))
        plt.clf()
        plt.close()


def plot_feature_elimination(feature_df: pd.DataFrame, dst_path: str):
    plt.figure(figsize=(text_width * cm * 0.5, text_width * cm * 0.45), dpi=dpi)
    feature_df["mean_test_score"] *= -1  # Invert scores to maximize
    plt.plot(feature_df.index, feature_df["mean_test_score"], marker='o', color='b', markersize=2)
    plt.fill_between(
        feature_df.index,
        feature_df['mean_test_score'] - feature_df['std_test_score'],
        feature_df['mean_test_score'] + feature_df['std_test_score'],
        color='lightblue', alpha=0.3,
    )

    n_features = feature_df["mean_test_score"].idxmin()
    plt.axvline(x=n_features, color="black", linestyle="--", label=f"Opt. Features: {n_features}")
    plt.title(f"Minimum MSE={feature_df['mean_test_score'].min():.2f}")
    plt.xlabel("Number of Features Selected")
    plt.ylabel("MSE")
    plt.legend()
    plt.ylim([0, 15])

    plt.tight_layout()
    plt.savefig(join(dst_path, "feature_analysis.png"), dpi=dpi)
    plt.clf()
    plt.close()


def plot_subject_correlations(df: pd.DataFrame, dst_path: str):
    metrics = []
    subjects = df["subject"].unique()
    for subject in df["subject"].unique():
        sub_df = df[df["subject"] == subject]
        metric, p_value = spearmanr(sub_df["ground_truth"], sub_df["prediction"])
        metrics.append(metric)

    fig, axs = plt.subplots(1, 1, figsize=(text_width * cm, text_width * cm * 0.5), dpi=dpi)
    axs.bar([f"{i + 1:2d}" for i in range(len(subjects))], metrics, color=primary_color)
    plt.ylim([0, 1])
    plt.title("Spearman's $\\rho$ Mean: {:.2f} Std: {:.2f}".format(np.mean(metrics), np.std(metrics)))
    plt.xlabel("Subjects")
    plt.ylabel("Spearman's $\\rho$")
    plt.tight_layout()

    if not exists(dst_path):
        makedirs(dst_path)

    plt.savefig(join(dst_path, "subject_performance.pdf"))
    plt.clf()
    plt.close()


def create_residual_plot(
        df: pd.DataFrame,
        log_path: str,
        file_name: str,
):
    fig = plt.figure(figsize=(text_width * cm * 0.5, text_width * cm * 0.5), dpi=dpi)
    ax = fig.add_subplot(111)

    subjects = df["subject"].unique()
    hsv_colors = [mcolors.hsv_to_rgb((i / len(subjects), 1, 1)) for i in range(len(subjects))]
    for idx, subject in enumerate(subjects):
        sub_df = df[df["subject"] == subject]
        ground_truth = sub_df.loc[:, "ground_truth"]
        prediction = sub_df.loc[:, "prediction"]
        differences = ground_truth - prediction
        plt.plot(ground_truth, differences, "o", color=hsv_colors[idx], markersize=2)

    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Ground Truth")
    plt.ylabel("Residual (Ground Truth - Prediction)")
    plt.tight_layout()
    plt.savefig(join(log_path, f"{file_name}_residual.png"))


def create_scatter_plot(
        df: pd.DataFrame,
        log_path: str,
        file_name: str,
        exp_name: str,
):
    ground_truth = df.loc[:, "ground_truth"]
    prediction = df.loc[:, "prediction"]

    slope, intercept, r_value, p_value_1, std_error_1 = linregress(ground_truth, prediction)
    rmse2 = np.sqrt(mean_squared_error(ground_truth, prediction))
    rmse = mean_squared_error(ground_truth, prediction, squared=False)
    r2 = r2_score(ground_truth, prediction)

    min_value = min(ground_truth.min(), prediction.min())
    max_value = max(ground_truth.max(), prediction.max())

    if exp_name in max_limits:
        max_value = max_limits[exp_name]

    min_value = int(min_value - min_value * 0.09)
    max_value = int(max_value + max_value * 0.09)

    x_values = np.arange(int(min_value * 100), int(max_value * 100 + 50)) / 100
    y_values = intercept + slope * x_values

    fig = plt.figure(figsize=(text_width * cm * 0.5, text_width * cm * 0.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(line_width)
    ax.spines["left"].set_linewidth(line_width)

    plt.plot(x_values, x_values, '-', color="gray", linewidth=line_width)
    plt.plot(x_values, y_values, '--', color=primary_color, linewidth=line_width)

    sign = "+" if intercept > 0 else ""
    margin = (max_value - min_value) * 0.95
    plt.text(
        min_value + margin,
        max_value - margin,
        f"$RMSE={rmse:.2f}$\n$R^{{2}}={r2:.2f}$\n$y={slope:.2f}x{sign}{intercept:.2f}$",
        style="italic",
        horizontalalignment="right",
    )

    # Create custom HSV color map
    subjects = df["subject"].unique()
    hsv_colors = [mcolors.hsv_to_rgb((i / len(subjects), 1, 1)) for i in range(len(subjects))]
    for idx, subject in enumerate(subjects):
        sub_df = df[df["subject"] == subject]
        plt.scatter(sub_df["ground_truth"], sub_df["prediction"], s=blob_size, color=hsv_colors[idx])

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

    plt.savefig(join(log_path, f"{file_name}_scatter.png"))
    plt.clf()
    plt.cla()
    plt.close()


def create_bland_altman_plot(
        df: pd.DataFrame,
        log_path: str,
        file_name: str,
        exp_name: str = "rpe",
        sd_limit: float = 1.96,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
):
    m1 = df.loc[:, "prediction"]
    m2 = df.loc[:, "ground_truth"]

    fig = plt.figure(figsize=(text_width * cm * 0.5, text_width * cm * 0.5), dpi=dpi)
    ax = fig.add_subplot(111)

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    subjects = df["subject"].unique()
    hsv_colors = [mcolors.hsv_to_rgb((i / len(subjects), 1, 1)) for i in range(len(subjects))]
    for idx, subject in enumerate(subjects):
        subject_mask = df["subject"] == subject
        ax.scatter(means[subject_mask], diffs[subject_mask], s=blob_size, color=hsv_colors[idx])

    # ax.scatter(means, diffs, s=blob_size, c=primary_color)  # , alpha=0.5)
    ax.axhline(mean_diff, **{"color": primary_color, "linewidth": 1, "linestyle": "--"})

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

    if exp_name != "rpe":
        plt.xlim(([0, 500]))
        plt.ylim(([-300, 500]))

    fig.tight_layout()

    if not exists(log_path):
        makedirs(log_path)

    plt.savefig(join(log_path, f"{file_name}_ba.png"))
    plt.clf()
    plt.cla()
    plt.close()


def create_model_performance_plot(result_df: pd.DataFrame, log_path: str, metric: str = "$R^{2}$"):
    plt.figure(figsize=(text_width * cm * 0.5, text_width * cm * 0.5), dpi=dpi)

    n_models = len(result_df["model"].unique())
    spacing = 0.25
    for idx, (model_name, model_df) in enumerate(result_df.groupby("model")):
        model_df.sort_values(by="temporal_context", inplace=True)
        x = model_df["temporal_context"].values
        x_jittered = x + (idx * spacing - (n_models - 1) * spacing / 2)

        y = model_df[f"{metric}_mean"].values
        e = model_df[f"{metric}_std"].values
        plt.errorbar(x_jittered, y, e, label=model_name, marker="o", capsize=3, markersize=2, alpha=1.0)

    plt.xticks([0, 6, 9, 12])
    plt.legend()
    plt.ylabel("$R^{2}$")
    plt.xlabel("Temporal Context")

    plt.tight_layout()
    plt.savefig(join(log_path, "model_performance.png"), dpi=dpi)
    plt.close()
