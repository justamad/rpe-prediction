import numpy as np
import logging
import matplotlib.pyplot as plt

from tensorflow import keras
from os.path import join
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


class PerformancePlotCallback(keras.callbacks.Callback):

    def __init__(self, train_gen, test_gen, log_path: str):
        super().__init__()
        self._train_gen = train_gen
        self._test_gen = test_gen
        self._log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Plotting performance... for epoch {epoch}")

        train_pred, train_labels, train_title = self.evaluate_for_generator(self._train_gen, training=True)
        train_n_pred, train_n_labels, train_n_title = self.evaluate_for_generator(self._train_gen)
        test_pred, test_labels, test_title = self.evaluate_for_generator(self._test_gen)

        fig, axs = plt.subplots(3, 1, sharey=True, figsize=(10, 10))
        axs[0].set_title("Train: " + train_title)
        axs[0].plot(train_pred, label="Predicted")
        axs[0].plot(train_labels, label="True")

        axs[1].set_title("Train No Dropout: " + train_n_title)
        axs[1].plot(train_n_pred, label="Predicted")
        axs[1].plot(train_n_labels, label="True")

        axs[2].set_title("Test: " + test_title)
        axs[2].plot(test_pred, label="Predicted")
        axs[2].plot(test_labels, label="True")

        plt.tight_layout()
        plt.savefig(join(self._log_path, f"{epoch:03d}.png"))
        plt.close()

    def evaluate_for_generator(self, generator, training: bool = False):
        predictions, labels = [], []
        for i in range(len(generator)):
            X_batch, y_batch = generator[i]
            pred = np.array(self.model(X_batch, training=training)).reshape(-1)
            predictions.extend(list(pred))
            labels.extend(list(y_batch.reshape(-1)))

        metrics = {
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "mape": mean_absolute_percentage_error,
            "r2": r2_score,
        }

        results = [f"{metric.upper()}: {metrics[metric](labels, predictions):.2f}" for metric in metrics.keys()]
        results = ", ".join(results)
        return predictions, labels, results
