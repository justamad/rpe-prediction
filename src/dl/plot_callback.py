import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from os.path import join
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr


class ProgressPlotCallback(keras.callbacks.Callback):

    def __init__(self, train_gen, test_gen, val_gen, log_path: str, gen_step: int = 5):
        super().__init__()
        self._train_gen = train_gen
        self._test_gen = test_gen
        self._val_gen = val_gen
        self._log_path = log_path
        self._gen_step = gen_step
        os.makedirs(self._log_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._gen_step != 0:
            return

        logging.info(f"Plotting performance... for epoch {epoch}")

        train_pred, train_labels, train_title = self.evaluate_for_generator(self._train_gen, training=True)
        train_n_pred, train_n_labels, train_n_title = self.evaluate_for_generator(self._train_gen)
        test_pred, test_labels, test_title = self.evaluate_for_generator(self._test_gen)
        val_pred, val_labels, val_title = self.evaluate_for_generator(self._val_gen)

        fig, axs = plt.subplots(4, 1, sharey=True, figsize=(10, 10))
        axs[0].set_title("Train: " + train_title)
        axs[0].plot(train_pred, label="Predicted")
        axs[0].plot(train_labels, label="True")

        axs[1].set_title("Train No Dropout: " + train_n_title)
        axs[1].plot(train_n_pred, label="Predicted")
        axs[1].plot(train_n_labels, label="True")

        axs[2].set_title("Test: " + test_title)
        axs[2].plot(test_pred, label="Predicted")
        axs[2].plot(test_labels, label="True")

        axs[3].set_title("Validation: " + val_title)
        axs[3].plot(val_pred, label="Predicted")
        axs[3].plot(val_labels, label="True")

        for i in range(4):
            axs[i].set_ylim([0, 21])

        plt.tight_layout()
        plt.savefig(join(self._log_path, f"{epoch:03d}.png"))
        plt.close()

    def evaluate_for_generator(self, generator, training: bool = False):
        predictions, labels = [], []

        if isinstance(generator, tf.data.Dataset):
            for X_batch, y_batch in generator:
                pred = np.array(self.model(X_batch, training=training)).reshape(-1)
                predictions.extend(list(pred))
                labels.extend(list(np.array(y_batch).reshape(-1)))
        else:
            for i in range(len(generator)):
                X_batch, y_batch = generator[i]
                if len(y_batch.shape) == 2:
                    y_batch = y_batch[:, 0]

                pred = np.array(self.model(X_batch, training=training))  # .reshape(-1)
                predictions.extend(list(pred))
                labels.extend(list(y_batch))  # .reshape(-1)))

        metrics = {
            "mse": lambda x, y: mean_squared_error(x, y, squared=True),
            "rmse": lambda x, y: mean_squared_error(x, y, squared=False),
            "mae": mean_absolute_error,
            "mape": mean_absolute_percentage_error,
            "r2": r2_score,
            "Rho": lambda x, y: spearmanr(x, y)[0],
        }

        results = [f"{metric.upper()}: {metrics[metric](labels, predictions):.2f}" for metric in metrics.keys()]
        results = ", ".join(results)
        return predictions, labels, results
