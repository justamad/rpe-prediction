from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import tensorflow as tf


def build_multi_branch_model(n_samples, n_features):
    """
    Builds a multi branch Tensorflow model
    :param n_samples: the number of samples
    :param n_features: number of features per column
    :return: final Tensorflow model
    """
    l2_factor = 0.05
    dropout_factor = 0.5
    filters = 64
    input = layers.Input(shape=(n_samples, n_features))
    imu = imu_part(input, filters, l2_factor, dropout_factor)
    skeleton = imu_part(input, filters, l2_factor, dropout_factor)

    # Branch together
    concat = layers.Concatenate([imu, skeleton])
    fc_1 = layers.Dense(30, activation="relu")(concat)
    fc_2 = layers.Dense(30, activation="relu")(fc_1)
    out = layers.Dense(1, activation=None)(fc_2)
    model = tf.keras.Model(inputs=[imu_part], outputs=[out])
    return model


def imu_part(input_layer, filters=64, l2_factor=0.05, dropout_factor=0.5):
    """
    Build the IMU branch of the network
    :param input_layer:
    :param filters:
    :param l2_factor:
    :param dropout_factor:
    :return: last conv layer
    """
    b1 = layers.Conv1D(filters=filters, kernel_size=8, padding='same', kernel_regularizer=l2(l2_factor))(input_layer)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation(activation='relu')(b1)
    b1 = layers.Dropout(dropout_factor)(b1)

    b2 = layers.Conv1D(filters=filters * 2, kernel_size=5, padding='same', kernel_regularizer=l2(l2_factor))(b1)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation(activation='relu')(b2)
    b2 = layers.Dropout(dropout_factor)(b2)

    b3 = layers.Conv1D(filters=filters, kernel_size=3, padding='same', kernel_regularizer=l2(l2_factor))(b2)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation(activation='relu')(b3)
    b3 = layers.Dropout(dropout_factor)(b3)
    return b3
