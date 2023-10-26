from os.path import join

import os
import json
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('WebAgg')
import plot_settings as ps
import matplotlib.pyplot as plt


subject_path = "/media/ch/Data/RPE_DATA_SET_PAPER"

rpe_heatmap = []
rpe_flat = []
for subject in os.listdir(subject_path):
    with open(join(subject_path, subject, "rpe_ratings.json"), "r") as f:
        data = json.load(f)

    cur_list = data["rpe_ratings"]
    rpe_flat.extend(cur_list)

    zero_fill = 12 - len(cur_list)
    cur_list = cur_list + [0] * zero_fill
    rpe_heatmap.append(cur_list)


def create_heatmap(data, mask):
    a = [i if i % 2 == 1 else "" for i in range(1, 13)]
    # colormap = sns.color_palette("Reds")
    cmap = sns.cm.rocket_r

    plt.figure(figsize=(ps.image_width * ps.cm, ps.image_width * ps.cm), dpi=300)
    ax = sns.heatmap(data, vmin=11, vmax=20, mask=mask, linewidth=0.5, yticklabels=a, xticklabels=a, cmap=cmap)
    plt.ylabel("Subject")
    plt.xlabel("Set")
    # plt.show()
    plt.tight_layout()
    plt.savefig("rpe_heatmap.pdf", dpi=300)
    plt.close()


def create_histogram(data):
    minimum = min(data)
    maximum = max(data)

    plt.figure(figsize=(ps.image_width * ps.cm, ps.image_width * ps.cm), dpi=300)
    counts, edges, bars = plt.hist(data, bins=range(minimum, maximum + 2))
    plt.xticks(np.arange(minimum, maximum + 1) + 0.5, np.arange(minimum, maximum + 1))
    plt.xlabel("RPE")
    plt.ylabel("Count")
    plt.bar_label(bars)
    plt.ylim(0, max(counts) + 3)
    # plt.show()
    plt.tight_layout()
    plt.savefig("rpe_histogram.pdf", dpi=300)
    plt.close()


rpe_heatmap = np.array(rpe_heatmap)
mask = rpe_heatmap == 0
create_heatmap(rpe_heatmap, mask)
create_histogram(rpe_flat)