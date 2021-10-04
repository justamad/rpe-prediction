from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import tensorflow as tf


def build_fcn_regression_model(
        seq_len: int,
        n_dim: int,
        nr_filter: int = 128,
        activation: str = None,
        l2_factor: float = 0.03,
        dropout: float = 0.5,
):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(seq_len, n_dim)))

    model.add(layers.Conv1D(filters=nr_filter, kernel_size=8, padding='same', kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation='relu'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv1D(filters=nr_filter * 2, kernel_size=5, padding='same', kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation='relu'))
    model.add(layers.Dropout(dropout))

    model.add(layers.Conv1D(filters=nr_filter, kernel_size=3, padding='same', kernel_regularizer=l2(l2_factor)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation='relu'))
    model.add(layers.Dropout(dropout))

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dense(25, activation='relu', kernel_regularizer=l2(l2_factor)))
    model.add(layers.Dense(1, activation=activation, kernel_regularizer=l2(l2_factor)))
    return model

# def build_fcn_classification_model(seq_len: int = 30, n_dim: int = 36, n_classes: int = 14):
#     l2_factor = 0.05
#     dropout_factor = 0.5
#     filters = 64
#
#     model = tf.keras.Sequential()
#     model.add(layers.Input(shape=(seq_len, n_dim)))
#
#     model.add(layers.Conv1D(filters=filters, kernel_size=8, padding='same', kernel_regularizer=l2(l2_factor)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation(activation='relu'))
#     model.add(layers.Dropout(dropout_factor))
#
#     model.add(layers.Conv1D(filters=filters * 2, kernel_size=5, padding='same', kernel_regularizer=l2(l2_factor)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation(activation='relu'))
#     model.add(layers.Dropout(dropout_factor))
#
#
#     model.add(layers.Conv1D(filters=filters, kernel_size=3, padding='same', kernel_regularizer=l2(l2_factor)))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Activation(activation='relu'))
#     model.add(layers.Dropout(dropout_factor))
#
#     model.add(layers.GlobalAveragePooling1D())
#     model.add(layers.Dense(25, activation='relu', kernel_regularizer=l2(l2_factor)))
#
#     if n_classes > 2:
#         model.add(layers.Dense(n_classes, activation='softmax', kernel_regularizer=l2(l2_factor)))
#     else:
#         model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_factor)))
#     return model
