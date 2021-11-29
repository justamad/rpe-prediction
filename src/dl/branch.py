from tensorflow.keras import layers, Model


def build_branch_model(seq_len_1: int, seq_len_2: int, n_dim: int) -> Model:
    # Kinect branch
    kinect_input = layers.Input(shape=(seq_len_1, 51, 1), name="b0_0")
    kinect_cnn_1 = layers.Conv2D(activation='relu', kernel_size=(3, 3), filters=n_dim, name='b0_1', strides=(1, 3))(kinect_input)
    kinect_cnn_2 = layers.Conv2D(activation='relu', kernel_size=(3, 3), filters=n_dim, name='b0_2')(kinect_cnn_1)
    kinect_cnn_3 = layers.Conv2D(activation='relu', kernel_size=(3, 3), filters=n_dim, name='b0_3')(kinect_cnn_2)

    kinect_flat = layers.Flatten()(kinect_cnn_3)

    kinect_fc_1 = layers.Dense(30, activation='relu')(kinect_flat)
    kinect_fc_2 = layers.Dense(30, activation='relu')(kinect_fc_1)

    # IMU branch
    imu_input = layers.Input(shape=(seq_len_2, 36, 1), name='b1_0')
    imu_cnn_1 = layers.Conv2D(activation='relu', kernel_size=(3, 3), filters=n_dim, name='b1_1', strides=(1, 3))(imu_input)
    imu_cnn_2 = layers.Conv2D(activation='relu', kernel_size=(3, 3), filters=n_dim, name='b1_2')(imu_cnn_1)
    imu_cnn_3 = layers.Conv2D(activation='relu', kernel_size=(3, 3), filters=n_dim, name='b1_3')(imu_cnn_2)

    imu_flat = layers.Flatten()(imu_cnn_3)

    imu_fc_1 = layers.Dense(30, activation='relu')(imu_flat)
    imu_fc_2 = layers.Dense(30, activation='relu')(imu_fc_1)

    # Concat Layers
    concat = layers.concatenate([kinect_fc_2, imu_fc_2])
    fc = layers.Dense(30, activation='relu')(concat)
    out = layers.Dense(1, activation=None)(fc)
    return Model(inputs=[kinect_input, imu_input], outputs=out)
