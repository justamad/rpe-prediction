from tensorflow import keras
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2
from keras.layers import (
    Input,
    BatchNormalization,
    GRU,
    Dropout,
    Activation,
    Dense,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Flatten,
)


def build_cnn_fc_model(hp, win_size, n_features):
    model = build_cnn_backbone(hp, win_size, n_features)
    model.add(Flatten())
    model.add(Dense(hp.Choice("fc_units_1", values=[32, 64, 128]), activation="relu"))
    model.add(Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5)))
    model.add(Dense(hp.Choice("fc_units_2", values=[32, 64, 128]), activation="relu"))
    model.add(Dropout(hp.Float("dropout", min_value=0.0, max_value=0.5)))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    return model


def build_cnn_lstm_model(hp, win_size, n_features):
    model = build_cnn_backbone(hp, win_size, n_features)
    model.add(LSTM(hp.Choice("gru_units", values=[32, 64, 128]), activation="tanh", return_sequences=False))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    return model


def build_cnn_gru_model(hp, win_size, n_features):
    model = build_cnn_backbone(hp, win_size, n_features)
    model.add(GRU(hp.Choice("gru_units", values=[32, 64, 128]), activation="tanh", return_sequences=False))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    return model


def build_cnn_backbone(hp, win_size, n_features):
    model = keras.Sequential()
    model.add(Input(shape=(win_size, n_features)))

    for i in range(hp.Choice('n_layers', values=[2, 3])):
        model.add(Conv1D(
            filters=hp.Choice(f"n_filters", values=[32, 64]) * (2 ** i),
            kernel_size=hp.Choice(f"kernel_size", values=[3, 7, 11]),
            padding="valid",
            activation=None,
            kernel_regularizer=l2(0.01),
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Activation("relu"))

    return model
