import tensorflow as tf

from typing import Tuple
from tensorflow import keras
from keras.layers import Input, Conv1D, Conv2D, BatchNormalization, GRU, Dropout, MaxPooling2D, Flatten, Dense, Reshape, Masking, GlobalAveragePooling2D, MaxPooling1D
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2


def build_conv2d_model(
        n_layers: int,
        n_filters: int,
        kernel_size: Tuple[int, int],
        dropout: float,
        n_units: int,
        learning_rate: float,
):
    _, n_samples, n_features, n_channels = (None, 170, 39, 3)
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, n_channels)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * (2 ** i), kernel_size=kernel_size, padding="valid", activation="relu",
                         kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(n_units, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(n_units, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=None))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()],
    )
    return model


def build_cnn_lstm_model(
        win_size: int,
        n_filters: int,
        kernel_size: Tuple[int, int],
        n_layers: int,
        dropout: float,
        lstm_units: int,
        learning_rate: float,
):
    _, n_samples, n_features, n_channel = (None, win_size, 39, 3)
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, n_channel)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * 2 ** i, kernel_size=kernel_size, padding="same", activation="relu",
                         kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        # model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))
    model.add(GRU(lstm_units, activation="relu", return_sequences=False))
    model.add(Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    return model
