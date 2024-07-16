import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .plot_settings import TEXT_WIDTH_INCH, DPI, LINE_WIDTH, BLOB_SIZE, HALF_PLOT_SIZE, CUT_OFF, get_colors
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr, linregress
from os.path import join, exists
from os import makedirs
from matplotlib.ticker import MaxNLocator, AutoLocator

y_labels = {"rpe": "RPE [Borg Scale]", "poweravg": "Power [Watts]", }
y_limits = {"rpe": (10, 21), "poweravg": (0, 300), }
max_limits = {"poweravg": 800}  # Used to equally scale physical model and ML models


def plot_sample_predictions(
        value_df: pd.DataFrame,
        exp_name: str,
        dst_path: str,
        pred_col: str = "prediction",
        label_col: str = "ground_truth"
):
    makedirs(dst_path, exist_ok=True)

    if exp_name not in y_labels or exp_name not in y_limits:
        raise ValueError(f"Unknown experiment '{exp_name}' for plotting predictions.")

    color1, color2 = get_colors(2)
    for subject_name in value_df["subject"].unique():
        subject_df = value_df[value_df["subject"] == subject_name]
        ground_truth = subject_df[label_col].values
        predictions = subject_df[pred_col].values

        r2 = r2_score(ground_truth, predictions)
        mape = mean_absolute_percentage_error(predictions, ground_truth)
        sp, _ = spearmanr(ground_truth, predictions)

        plt.figure(figsize=(HALF_PLOT_SIZE, HALF_PLOT_SIZE), dpi=DPI)
        plt.plot(ground_truth, label="Ground Truth", color=color1)
        plt.plot(predictions, label="Prediction", color=color2)
        plt.ylim(y_limits[exp_name])
        plt.title(f"$R^2$={r2:.2f}, MAPE={mape:.2f}, $\\rho$={sp:.2f}")
        plt.xlabel("Repetition")
        plt.ylabel(y_labels[exp_name])
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(dst_path, f"{subject_name}.pdf"), bbox_inches='tight', pad_inches=CUT_OFF, dpi=DPI)
        plt.clf()
        plt.close()


def plot_feature_elimination(feature_df: pd.DataFrame, dst_path: str):
    plt.figure(figsize=(HALF_PLOT_SIZE, HALF_PLOT_SIZE), dpi=DPI)
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
    plt.savefig(join(dst_path, "feature_analysis.png"), dpi=DPI)
    plt.clf()
    plt.close()


def plot_subject_correlations(df: pd.DataFrame, dst_path: str):
    metrics = []
    subjects = df["subject"].unique()
    for subject in df["subject"].unique():
        sub_df = df[df["subject"] == subject]
        metric, p_value = spearmanr(sub_df["ground_truth"], sub_df["prediction"])
        metrics.append(metric)

    color = get_colors(1)
    fig, axs = plt.subplots(1, 1, figsize=(TEXT_WIDTH_CM, HALF_PLOT_SIZE), dpi=DPI)
    axs.bar([f"{i + 1:2d}" for i in range(len(subjects))], metrics, color=color[0])
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
    plt.figure(figsize=(HALF_PLOT_SIZE, HALF_PLOT_SIZE), dpi=DPI)
    subjects = df["subject"].unique()
    for idx, (subject, color) in enumerate(zip(subjects, get_colors(len(subjects)))):
        sub_df = df[df["subject"] == subject]
        ground_truth = sub_df.loc[:, "ground_truth"]
        prediction = sub_df.loc[:, "prediction"]
        differences = ground_truth - prediction
        plt.plot(ground_truth, differences, "o", color=color, markersize=2)

    plt.ylim(-7.5, 7.5)
    plt.axhline(y=0, color="black", linestyle="--")
    plt.xlabel("Ground Truth")
    plt.ylabel("Residual (Ground Truth - Prediction)")
    plt.tight_layout()
    plt.savefig(join(log_path, f"{file_name}_residual.png"), bbox_inches='tight', pad_inches=CUT_OFF, dpi=DPI)


def create_scatter_plot(
        df: pd.DataFrame,
        log_path: str,
        file_name: str,
        exp_name: str,
):
    ground_truth = df.loc[:, "ground_truth"]
    prediction = df.loc[:, "prediction"]
    color = get_colors(1)[0]

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

    fig = plt.figure(figsize=(HALF_PLOT_SIZE, HALF_PLOT_SIZE), dpi=DPI)
    ax = fig.add_subplot(111)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(LINE_WIDTH)
    ax.spines["left"].set_linewidth(LINE_WIDTH)

    plt.plot(x_values, x_values, '-', color="gray", linewidth=LINE_WIDTH)
    plt.plot(x_values, y_values, '--', color=color, linewidth=LINE_WIDTH)

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
    for idx, (subject, color) in enumerate(zip(subjects, get_colors(len(subjects)))):
        sub_df = df[df["subject"] == subject]
        plt.scatter(sub_df["ground_truth"], sub_df["prediction"], s=BLOB_SIZE, color=color)

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

    plt.savefig(join(log_path, f"{file_name}_scatter.png"), bbox_inches='tight', pad_inches=CUT_OFF, dpi=DPI)
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

    fig = plt.figure(figsize=(HALF_PLOT_SIZE, HALF_PLOT_SIZE), dpi=DPI)
    ax = fig.add_subplot(111)

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    subjects = df["subject"].unique()
    hsv_colors = [mcolors.hsv_to_rgb((i / len(subjects), 1, 1)) for i in range(len(subjects))]
    for idx, subject in enumerate(subjects):
        subject_mask = df["subject"] == subject
        ax.scatter(means[subject_mask], diffs[subject_mask], s=BLOB_SIZE, color=hsv_colors[idx])

    color = get_colors(1)[0]
    # ax.scatter(means, diffs, s=blob_size, c=primary_color)  # , alpha=0.5)
    ax.axhline(mean_diff, **{"color": color, "linewidth": 1, "linestyle": "--"})

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


def create_model_performance_plot(df: pd.DataFrame, log_path: str, exp_name: str, metric: str, alt_name: str = None):
    limits = {"MSE": (0, 16), "RMSE": (0, 4), "MAE": (0, 4), "MAPE": (0, 20), "$R^{2}$": (-5, 1),
              "Spearman's $\\rho$": (-0.2, 1.05)}

    min_max = {"MSE": "min", "RMSE": "min", "MAE": "min", "MAPE": "min", "$R^{2}$": "max", "Spearman's $\\rho$": "max"}
    yticks = {"Spearman's $\\rho$": [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]}

    plt.figure(figsize=(HALF_PLOT_SIZE, HALF_PLOT_SIZE), dpi=DPI)
    n_models = len(df["model"].unique())
    spacing = 0.25

    for idx, (model_name, model_df) in enumerate(sorted(df.groupby("model"))):
        model_df.sort_values(by="temporal_context", inplace=True)
        x = model_df["temporal_context"].values
        x = x + (idx * spacing - (n_models - 1) * spacing / 2)

        y = model_df[f"{metric}_mean"].values
        e = model_df[f"{metric}_std"].values
        plt.errorbar(x, y, e, label=model_name, marker="o", capsize=3, markersize=2, alpha=1.0)

    plt.xticks([0, 6, 9, 12])

    if metric in yticks:
        plt.yticks(yticks[metric])

    plt.ylim(limits[metric])
    plt.legend()
    plt.ylabel(f"{metric}")
    plt.xlabel("Temporal Context")
    best_score = df[f"{metric}_mean"].min() if min_max[metric] == "min" else df[f"{metric}_mean"].max()
    plt.title("Top Score: {:.2f}".format(best_score))

    plt.tight_layout()
    plt.savefig(join(log_path, f"{exp_name}_{metric if alt_name is None else alt_name}.png"), dpi=DPI)
    plt.close()


def create_correlation_heatmap(df: pd.DataFrame, file_name: str):
    plt.figure(figsize=(TEXT_WIDTH_INCH, TEXT_WIDTH_INCH * 0.8), dpi=DPI)
    cbar_kws = {"label": "Pearson's Correlation Coefficient", }

    cmap = mpl.colormaps.get_cmap('brg_r')
    sns.heatmap(df, fmt=".2f", vmin=-1, vmax=1, cmap=cmap, linewidth=0.5, annot=True, cbar_kws=cbar_kws)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight", pad_inches=CUT_OFF, dpi=DPI)
    plt.close()
