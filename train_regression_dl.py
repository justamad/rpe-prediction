from rpe_prediction.dl import build_fcn_regression_model
from rpe_prediction.features import prepare_data_for_deep_learning
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
parser.add_argument('--n_features', type=int, dest='n_features', default=60)
parser.add_argument('--n_frames', type=int, dest='n_frames', default=104)
parser.add_argument('--n_filters', type=int, dest='n_filters', default=128)
args = parser.parse_args()

model = build_fcn_regression_model(args.n_frames, args.n_features, args.n_filters, 'relu')


def create_folder_if_not_already_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


base_path = join(args.result_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), "fcn")
create_folder_if_not_already_exists(base_path)

X, y = prepare_data_for_deep_learning(args.src_path)

train_x, train_y = None, None
valid_x, valid_y = None, None

print(f'Training dimensions: {train_x.shape}, {train_y.shape}')
print(f'Testing dimension {valid_x.shape}, {valid_y.shape}')

model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(lr=0.001),
              metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.summary()

# Train the network
t = datetime.now()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
checkpoint_filepath = join(base_path, 'checkpoints/')
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                      monitor='val_loss', mode='min', save_best_only=True)

# model.fit_generator()
history = model.fit(train_inputs, train_y, batch_size=32, epochs=1000, verbose=1,
                    validation_data=(valid_inputs, valid_y), callbacks=[early_stopping, model_checkpoint, reduce_lr])

print(f'Training time: {datetime.now() - t}')

# Plot the results
plt.figure(1)
plt.plot(history.history['loss'], 'b', label='Training Loss')
plt.title('Training Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
plt.xlabel("Iterations")
plt.ylabel("Binary Crossentropy")
plt.legend()
plt.tight_layout()
plt.savefig(join(base_path, "history.png"))
plt.show()

# Print the minimum loss
print("Training loss", np.min(history.history['loss']))
print("Validation loss", np.min(history.history['val_loss']))

# Save trained model to file
model_name = join(base_path, "models")
model.save(model_name)
