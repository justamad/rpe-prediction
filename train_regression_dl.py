from src.ml import split_data_based_on_pseudonyms_multiple_inputs
from src.dl import TimeSeriesGenerator, build_branch_model
from src.utils import create_folder_if_not_already_exists
from src.features import collect_all_trials_with_labels

from argparse import ArgumentParser
from os.path import join
from datetime import datetime

import tensorflow as tf
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M:%S',
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('my_logger').addHandler(console)

parser = ArgumentParser()
parser.add_argument('--src_path', type=str, dest='src_path', default="data/processed")
parser.add_argument('--result_path', type=str, dest='result_path', default="results")
parser.add_argument('--n_features', type=int, dest='n_features', default=51)
parser.add_argument('--n_frames', type=int, dest='n_frames', default=60)
parser.add_argument('--n_filters', type=int, dest='n_filters', default=128)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=32)
parser.add_argument('--epochs', type=int, dest='epochs', default=10)
args = parser.parse_args()

base_path = join(args.result_path, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
create_folder_if_not_already_exists(base_path)

X, y = collect_all_trials_with_labels(args.src_path)
X_train, y_train, X_test, y_test = split_data_based_on_pseudonyms_multiple_inputs(X, y, train_p=0.75, random_seed=69)
X_train, y_train, X_val, y_val = split_data_based_on_pseudonyms_multiple_inputs(X_train, y_train, train_p=0.75, random_seed=69)

train_gen = TimeSeriesGenerator(
    X_train,
    y_train,
    batch_size=args.batch_size,
    win_size=2,
    overlap=0.9,
)

val_gen = TimeSeriesGenerator(
    X_val,
    y_val,
    batch_size=args.batch_size,
    win_size=2,
    overlap=0.9,
)

test_gen = TimeSeriesGenerator(
    X_test,
    y_test,
    batch_size=args.batch_size,
    win_size=2,
    overlap=0.9,
    shuffle=False,
    balance=False,
)

model = build_branch_model(seq_len_1=train_gen.get_x1_dim, seq_len_2=train_gen.get_x2_dim, n_filters=args.n_filters)

model.compile(
    loss='mean_squared_error',
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()],
)
model.summary()

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau()
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=join(base_path, 'checkpoints/'),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=join(base_path, "logs"),
    histogram_freq=1,
)


history = model.fit(
    train_gen,
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
plt.plot(history.history['loss'], 'b', label='Training Loss')
plt.title('Training Loss')
plt.plot(history.history['val_loss'], 'r', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig(join(base_path, "history.png"))
plt.show()
