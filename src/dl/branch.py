from tensorflow.keras import layers, Model


def build_branch_model(
        seq_len_1: int,
        seq_len_2: int,
        n_filters: int,
) -> Model:
    kinect_input = layers.Input(shape=(seq_len_1, 54), name="b0_0")
    kinect_cnn_1 = layers.Conv1D(activation='relu', kernel_size=10, filters=n_filters, name='b0_1')(kinect_input)
    kinect_bn_1 = layers.BatchNormalization()(kinect_cnn_1)
    kinect_cnn_2 = layers.Conv1D(activation='relu', kernel_size=10, filters=n_filters * 2, name='b0_2')(kinect_bn_1)
    kinect_do_1 = layers.Dropout(rate=0.2)(kinect_cnn_2)

    imu_input = layers.Input(shape=(seq_len_2, 36), name='b1_0')
    imu_cnn_1 = layers.Conv1D(activation='relu', kernel_size=28, filters=n_filters, name='b1_1', strides=2)(imu_input)
    imu_bn_1 = layers.BatchNormalization()(imu_cnn_1)
    imu_cnn_2 = layers.Conv1D(activation='relu', kernel_size=28, filters=n_filters * 2, name='b1_2', strides=2)(imu_bn_1)
    imu_do_1 = layers.Dropout(rate=0.2)(imu_cnn_2)

    fusion = layers.concatenate([kinect_do_1, imu_do_1])
    lstm_1 = layers.LSTM(4, activation='relu', return_sequences=True)(fusion)
    lstm_2 = layers.LSTM(4, activation='relu')(lstm_1)

    fc = layers.Dense(30, activation='relu')(lstm_2)
    out = layers.Dense(1, activation=None)(fc)
    return Model(inputs=[kinect_input, imu_input], outputs=out)
