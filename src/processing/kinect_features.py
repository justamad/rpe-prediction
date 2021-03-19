import pandas as pd
import numpy as np


def calculate_velocity(df: pd.DataFrame):
    data = df.to_numpy()
    velocity = np.gradient(data, axis=0)
    velocity_mean = velocity.mean(axis=0).reshape(1, 96)
    print(velocity_mean.shape)
    return pd.DataFrame(data=velocity_mean, columns=df.columns)


