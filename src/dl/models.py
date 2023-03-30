from tensorflow import keras
from keras import layers
from typing import Dict, Any


def build_conv1d_lstm_regression_model(
        n_filters: int,
        kernel_size: int,
        meta: Dict[str, Any],
        dropout: float = 0.5,
):
    _, n_samples, n_features = meta["X_shape_"]
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_samples, n_features)))

    model.add(layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding="same", ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv1D(filters=n_filters * 2, kernel_size=kernel_size, padding="same", ))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.GRU(units=4, return_sequences=False))
    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mse", "mae", "mape"])
    return model
