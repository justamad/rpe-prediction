import numpy as np
import pandas as pd
import cv2

from src.dataset import dl_normalize_data_3d_subject, zero_pad_dataset

X = np.load("data/training/X_seg.npz", allow_pickle=True)["X"]
y = pd.read_csv("data/training/y_seg.csv", index_col=0)
X = dl_normalize_data_3d_subject(X, y, method="min_max") * 255
X = zero_pad_dataset(X, 170)
# X = np.load("data/training/X_lstm.npz", allow_pickle=True)["X"]
# y = pd.read_csv("data/training/y_lstm.csv", index_col=0)
# X = dl_normalize_data_3d_subject(X, y, method="min_max") * 255

for i in range(len(X)):
    image = X[i]
    image = image[:100, :, :]
    label = y.iloc[i]
    subject = label["subject"]
    set_id = label["set_id"]

    print(X.shape)
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    print(image.shape)
    print(image.min())
    print(image.max())

    cv2.imwrite(f"images/{subject}_{set_id}_{i}.png", image)
