from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np  # for som math operations
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/927394/00_imu.csv", delimiter=';', index_col="sensorTimestamp")
df = df[[c for c in df.columns if "ACCELERATION" in c]]

X = df.values
sc = StandardScaler()  # creating a StandardScaler object
X_std = sc.fit_transform(X)  # standardizing the data

pca = PCA()
X_pca = pca.fit(X_std)

# print(X_pca)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()

pca = PCA(n_components=0.99)
X_pca = pca.fit_transform(X_std)  # this will fit and reduce dimensions
print(pca.n_components_)

# df = pd.DataFrame(pca.components_, columns=df.columns)
# print(df)

n_pcs = pca.n_components_  # get number of component
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = df.columns
# get the most important feature names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print(most_important_names)
