from typing import Tuple
from tensorflow import keras
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    GRU,
    Dropout,
    MaxPooling2D,
    Flatten,
    Dense,
    Reshape,
    Conv1D,
    MaxPooling1D,
    TimeDistributed,
)


def build_autoencoder(
        n_layers: int,
        n_filters: int,
        kernel_size: Tuple[int, int],
        dropout: float,
        n_units: int,
        win_size: int,
        learning_rate: float,
):
    _, n_samples, n_features, n_channels = (None, win_size, 51, 1)
    model = keras.Sequential()

    model.add(Input(shape=n_features))
    model.add(Dense(n_filters, activation="relu"))
    model.add(Dense(n_filters / 2, activation="relu"))
    model.add(Dense(n_filters / 4, activation="relu"))
    model.add(Dense(n_filters / 2, activation="relu"))
    model.add(Dense(n_filters, activation="relu"))
    model.add(Dense(n_features, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mse", "mae", "mape"],
    )
    model.summary()
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
    _, n_samples, n_features, n_channel = (None, win_size, 51, 1)
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features)))

    # for i in range(n_layers):
    #     model.add(Conv1D(filters=n_filters * (2 ** i), kernel_size=3, padding="valid", activation="relu", kernel_regularizer=l2(0.01)))
    #     model.add(BatchNormalization())
    #     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(TimeDistributed(Dense(128, activation="relu")))
    model.add(TimeDistributed(Dense(64, activation="relu")))
    model.add(TimeDistributed(Dense(32, activation="relu")))

    model.add(Reshape((model.output_shape[1], model.output_shape[2])))  # * model.output_shape[3])))
    model.add(GRU(4, activation="relu", return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    model.summary()
    return model
