from tensorflow import keras
from tensorflow_addons.metrics import RSquare
from keras.regularizers import l2
from keras.layers import (
    Input,
    BatchNormalization,
    GRU,
    Dropout,
    Dense,
    Conv1D,
    MaxPooling1D,
    concatenate,
)

def build_dual_cnn_lstm_model(hp, win_size: int, n_features: int):
    input1 = Input(shape=(win_size, n_features,))
    input2 = Input(shape=(win_size, n_features,))
    model = concatenate([input1, input2])

    for i in range(hp.Choice('n_layers', values=[2, 3])):
        model.add(Conv1D(
            filters=hp.Choice('filters_1', values=[32, 64]) * (2 ** i),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 7, 11]),
            padding="valid",
            activation="relu",
            kernel_regularizer=l2(0.01),
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

    model.add(GRU(hp.Choice("gru_units", values=[8, 16, 32, 64, 128]), activation="tanh", return_sequences=False))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    # model.summary()
    return model


def build_cnn_lstm_model(hp, win_size, n_features):
    model = keras.Sequential()
    model.add(Input(shape=(win_size, n_features)))

    for i in range(hp.Choice('n_layers', values=[2, 3])):
        model.add(Conv1D(
            filters=hp.Choice('filters_1', values=[32, 64]) * (2 ** i),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 7, 11]),
            padding="valid",
            activation="relu",
            kernel_regularizer=l2(0.01),
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))

    # model.add(Reshape((model.output_shape[1], model.output_shape[2]))) #  * model.output_shape[3])))
    model.add(GRU(hp.Choice("gru_units", values=[8, 16, 32, 64, 128]), activation="tanh", return_sequences=False))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)),
        loss="mse", metrics=["mse", "mae", "mape", RSquare()]
    )
    # model.summary()
    return model
