from keras import layers
from keras.regularizers import l2

import tensorflow as tf


def build_fcn_regression_model(
        n_samples: int,
        n_features: int,
        nr_filter: int = 128,
        activation: str = None,
        l2_factor: float = 0.03,
        dropout: float = 0.5,
):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(n_samples, n_features)))

    model.add(layers.Conv1D(filters=nr_filter, kernel_size=64, padding="same", kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv1D(filters=nr_filter * 2, kernel_size=32, padding="same", kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv1D(filters=nr_filter, kernel_size=16, padding="same", kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(32, activation="relu", kernel_regularizer=l2(l2_factor)))
    model.add(layers.Dense(1, activation=activation, kernel_regularizer=l2(l2_factor)))
    return model


def build_conv_lstm_regression_model(
        n_samples: int,
        n_features: int,
        n_filters: int = 64,
        kernel_size: int = 32,
        # l2_factor: float = 0.03,
        dropout: float = 0.5,
):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(n_samples, n_features)))

    model.add(layers.Conv1D(filters=n_filters, kernel_size=kernel_size, padding="same", ))  # kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv1D(filters=n_filters * 2, kernel_size=kernel_size // 2, padding="same", ))  # kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation="relu"))
    model.add(layers.Dropout(dropout))

    model.add(layers.GRU(units=n_filters, return_sequences=False))
    model.add(layers.Dense(1))  # , kernel_regularizer=l2(l2_factor)))
    return model
