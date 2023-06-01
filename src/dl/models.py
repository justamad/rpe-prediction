import tensorflow as tf

from typing import Dict, Any, Tuple
from tensorflow import keras
from keras.layers import Input, Conv1D, Conv2D, BatchNormalization, Activation, GRU, Dropout, MaxPooling2D, Flatten, Dense, Reshape, Masking, GlobalAveragePooling2D, MaxPooling1D
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2


def build_conv2d_model(
        # meta: Dict[str, Any],
        increase_layers: bool = True,
        n_layers: int = 3,
        n_filters: int = 32,
        kernel_size: Tuple[int, int] = (3, 3),
        dropout: float = 0.5,
        n_units: int = 128,
        learning_rate: float = 1e-4,
):
    _, n_samples, n_features, n_channels = (1991, 170, 39, 3)
    # _, n_samples, n_features, n_channels = meta["X_shape_"]
    # n_outputs = meta["n_outputs_"]
    n_outputs = 1
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, n_channels)))

    for i in range(n_layers):
        n_filters = n_filters * 2 ** i if increase_layers else n_filters // 2 ** i
        model.add(Conv2D(filters=n_filters, kernel_size=kernel_size, padding="valid", activation="relu")) # , kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_units, activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()],
    )
    # model.summary()
    return model


def build_cnn_lstm_model(
        meta: Dict[str, Any],
        n_filters: int,
        kernel_size: Tuple[int, int],
        n_layers: int,
        dropout: float,
        lstm_units: int,
):
    _, n_samples, n_features, n_channel = meta["X_shape_"]
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, n_channel)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * 2 ** i, kernel_size=kernel_size, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        # model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    # model.add(Reshape((model.output_shape[1], model.output_shape[2] * model.output_shape[3])))
    # model.add(GRU(lstm_units, activation="relu", return_sequences=False))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse", "mae", "mape", RSquare()])
    return model
