from typing import Dict, Any, Tuple
from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, GRU, Dropout, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2


def build_conv_model(
        meta: Dict[str, Any],
        n_layers: int = 3,
        n_filters: int = 32,
        kernel_size: Tuple[int, int] = (10, 3),
        dropout: float = 0.3,
        n_units: int = 128,
):
    _, n_samples, n_features = meta["X_shape_"]
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, 1)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * (i + 1), kernel_size=kernel_size, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(n_units, activation="relu"))
    model.add(Dense(meta["n_outputs_"]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="mse", metrics=["mse", "mae", "mape", RSquare()], )
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
    model = keras.Sequential()
    model.add(Input(shape=(n_samples, n_features, 1)))

    for i in range(n_layers):
        model.add(Conv2D(filters=n_filters * (i + 1), kernel_size=kernel_size, padding="same", activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Reshape((model.output_shape[1], -1)))
    model.add(GRU(lstm_units, activation="relu"))
    model.add(Dense(meta["n_outputs_"]))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse", "mae", "mape"])
    return model
