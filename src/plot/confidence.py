import pandas as pd
import matplotlib.pyplot as plt


def plot_confidence_values(file_name, path=None):
    df = pd.read_csv(file_name, sep=';', index_col=False).rename(lambda c: c[:-4], axis='columns')
    df_s = df.sum(axis=0).transpose()
    df = df / df_s

    df.transpose().plot.barh(rot=0, stacked=True, figsize=(20, 10))
    plt.title("Confidence Values. 0=None (Out of range), 1=Low, 2=Medium.")
    plt.xlabel("Percentage")

    if path is None:
        plt.show()
    else:
        plt.savefig(path)

    plt.close()
