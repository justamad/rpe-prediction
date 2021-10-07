from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from rpe_prediction.dl import build_fcn_regression_model
from rpe_prediction.features import collect_all_trials_with_labels
from rpe_prediction.ml import split_data_based_on_pseudonyms
from argparse import ArgumentParser
from os.path import join
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--result_path', type=str, dest='result_path', default="results")
parser.add_argument('--n_features', type=int, dest='n_features', default=51)
parser.add_argument('--n_frames', type=int, dest='n_frames', default=60)
parser.add_argument('--n_filters', type=int, dest='n_filters', default=128)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
parser.add_argument('--epochs', type=int, dest='epochs', default=1)
args = parser.parse_args()

model = build_fcn_regression_model(args.n_frames, args.n_features, args.n_filters, 'relu')


def create_folder_if_not_already_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


base_path = join(args.result_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
create_folder_if_not_already_exists(base_path)

X, y = collect_all_trials_with_labels(args.src_path)
X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms(X, y, train_p=0.7, random_seed=42)
X_train, y_train, X_val, y_val = split_data_based_on_pseudonyms(X_train, y_train, train_p=0.7, random_seed=42)

print(f'Training dimensions: {X_train.shape}, {y_train.shape}')
print(f'Validation dimensions: {X_val.shape}, {y_val.shape}')
print(f'Testing dimension {X_test.shape}, {y_test.shape}')

model.compile(loss='mean_squared_error',
              optimizer=tf.optimizers.Adam(lr=0.001),
              metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],)
model.summary()

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=join(base_path, 'checkpoints/'),
                                                      save_weights_only=True,
                                                      monitor='val_loss',
                                                      mode='min',
                                                      save_best_only=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=join(base_path, "logs"),
                                                      histogram_freq=1)

train_gen = TimeseriesGenerator(X_train.to_numpy(),
                                y_train['rpe'].to_numpy(),
                                length=args.n_frames,
                                batch_size=args.batch_size,
                                shuffle=True)

val_gen = TimeseriesGenerator(X_val.to_numpy(),
                              y_val['rpe'].to_numpy(),
                              length=args.n_frames,
                              batch_size=args.batch_size,
                              shuffle=True)

history = model.fit(train_gen,
                    epochs=args.epochs,
                    verbose=1,
                    validation_data=val_gen,
                    # callbacks=[early_stopping, model_checkpoint, reduce_lr, tensorboard_callback],
                    callbacks=[model_checkpoint, tensorboard_callback],
                    )

# Save trained model to file
model_name = join(base_path, "models")
model.save(model_name)

# Plot the results
plt.figure(1)
plt.plot(history.history['loss'], 'b', label='Training Loss')
plt.title('Training Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig(join(base_path, "history.png"))
plt.show()
