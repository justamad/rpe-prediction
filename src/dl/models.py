from typing import Dict, Any, Tuple
from tensorflow import keras
from keras.layers import Input, Conv1D, Conv2D, BatchNormalization, GRU, Dropout, MaxPooling2D, Flatten, Dense, Reshape, GlobalMaxPooling2D, GlobalAveragePooling1D, Masking, GlobalAveragePooling2D
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2


def build_conv1d_model(
        meta: Dict[str, Any],
        n_layers: int = 3,
        n_filters: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.5,
        n_units: int = 128,
):
    _, n_samples, n_features, channels = meta["X_shape_"]
    _, n_outputs = meta["n_outputs_"]
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, channels)))
    model.add(Masking(mask_value=0.0))

    for i in range(n_layers):
        model.add(Conv1D(filters=n_filters * 2 ** i, kernel_size=kernel_size, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(GlobalMaxPooling2D())
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    model.add(Dense(n_units, activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mse", "mae", "mape", RSquare()], )
    return model


def build_conv2d_model(
        meta: Dict[str, Any],
        n_layers: int = 3,
        n_filters: int = 32,
        kernel_size: Tuple[int, int] = (10, 3),
        dropout: float = 0.5,
        n_units: int = 128,
):
    _, n_samples, n_features = meta["X_shape_"]
    _, n_outputs = meta["n_outputs_"]
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, 1)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * 2 ** i, kernel_size=kernel_size, padding="valid", activation="relu", kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(n_units, activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse", "mae", "mape", RSquare()], )
    return model


def build_cnn_lstm_model(
        meta: Dict[str, Any],
        n_filters: int = 32,
        kernel_size: Tuple[int, int] = (10, 3),
        n_layers: int = 3,
        dropout: float = 0.3,
        lstm_units: int = 32,
):
    _, n_samples, n_features = meta["X_shape_"]
    _, n_outputs = meta["n_outputs_"]
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, 1)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * 2 ** i, kernel_size=kernel_size, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Reshape((model.output_shape[1], -1)))
    model.add(GRU(lstm_units, activation="relu"))
    model.add(Dense(n_outputs))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse", "mae", "mape", RSquare()])
    return model
