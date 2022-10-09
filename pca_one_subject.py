from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np  # for som math operations
from sklearn.preprocessing import StandardScaler  # for standardizing the Data
from sklearn.decomposition import PCA  # for PCA calculation
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt  # for plotting
import json

subject = "AEBA3A"

json_file = f"data/processed/{subject}/rpe_ratings.json"
with open(json_file) as f:
    rpe_values = json.load(f)
    rpe_values = np.array(rpe_values["rpe_ratings"])


df = pd.DataFrame()
for i in range(12):
    try:
        cur_df = pd.read_csv(f"data/processed/{subject}/{i:02d}_imu.csv", delimiter=';', index_col="sensorTimestamp")
        cur_df["rpe"] = rpe_values[i]
        df = pd.concat([df, cur_df], ignore_index=True)
    except Exception as e:
        print(e)
        pass

# x = df.loc[:, listend1].values
# y= df.loc[:, 'Brain_Region'].values

plt.plot(df.loc[:, "CHEST_ACCELERATION_X"], label="X")
plt.plot(df.loc[:, "CHEST_ACCELERATION_Y"], label="Y")
plt.plot(df.loc[:, "CHEST_ACCELERATION_Z"], label="Z")
plt.legend()
plt.show()

y = df.loc[:, 'rpe'].values
df = df.drop(columns=['rpe'])
df = df[[c for c in df.columns if "ACCELERATION" in c]]
x = df.values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

n_pcs = pca.n_components_  # get number of component
# get the index of the most important feature on EACH component
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = df.columns
# get the most important feature names
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
print(most_important_names)


principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])
principalDf["rpe"] = y

# finalDf = pd.concat([principalDf, df[['Brain_Region']]], axis=1)
plt.scatter(x=principalDf["pc1"], y=principalDf["pc2"], c=principalDf['rpe'], cmap='viridis', alpha=0.2, linewidths=0)
plt.show()
