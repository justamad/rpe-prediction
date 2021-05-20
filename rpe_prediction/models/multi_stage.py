from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import tensorflow as tf


def build_multi_branch_model(n_samples, n_features, l2_factor=0.05, dropout=0.5, n_filters=64):

    imu_input = layers.Input(shape=(n_samples, n_features))
    skeleton_input = layers.Input(shape=(n_samples, n_features))

    imu = build_imu_branch(imu_input, n_filters, l2_factor, dropout)
    skeleton = build_imu_branch(skeleton_input, n_filters, l2_factor, dropout)

    # Merge different branches together
    concat = layers.concatenate([imu, skeleton], axis=-1)
    lstm = layers.LSTM(128, activation='tanh')(concat)
    fc_1 = layers.Dense(30, activation='relu')(lstm)
    fc_2 = layers.Dense(30, activation='relu')(fc_1)
    out = layers.Dense(1, activation=None)(fc_2)
    return tf.keras.Model(inputs=[imu_input, skeleton_input], outputs=[out])


def build_imu_branch(input_layer, filters=64, l2_factor=0.05, dropout_factor=0.5):
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


def build_skeleton_branch(input_layer, filters=64, l2_factor=0.05, dropout_factor=0.5):
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


if __name__ == '__main__':
    model = build_multi_branch_model(30, 92)
