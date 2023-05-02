from .plot_settings import column_width, dpi, cm
from os.path import join, exists
from os import makedirs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

    fig = plt.figure(figsize=(column_width * cm, column_width * cm), dpi=dpi)
    ax = fig.add_subplot(111)
    # sm.graphics.mean_diff_plot(m1, m2, ax=ax)

    means = np.mean([m1, m2], axis=0)
    diffs = m1 - m2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, axis=0)

    # Plot individual gait speed colors
    ax.scatter(means, diffs, s=0.1)
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
                    # fontsize=FONT_SIZE,
                    xycoords='axes fraction')
        ax.annotate(f'+{sd_limit:.2f} SD: {upper:.2f}',
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    # fontsize=FONT_SIZE,
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
