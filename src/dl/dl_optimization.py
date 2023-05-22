import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from src.ml import MLOptimization, LearningModelBase
from typing import Union, List
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from os import makedirs
from os.path import join


class DLOptimization(MLOptimization):

    def __init__(self, X: Union[pd.DataFrame, np.ndarray], y: pd.DataFrame, balance: bool, task: str, mode: str,
                 ground_truth: Union[str, List[str]], n_splits: int = None):
        super().__init__(X, y, balance, ground_truth, task, mode, n_splits)

    def perform_grid_search_with_cv(
            self,
            model_config: LearningModelBase,
            log_path: str,
            view_progress: int = 1,
            verbose: int = 1,
            patience: int = 3,
    ):
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        avg_score = SubjectScoresAvg()
        folder = join(log_path, str(model_config), timestamp)

        for combination in ParameterGrid(model_config.parameters):
            batch_size = combination["batch_size"]
            epochs = combination["epochs"]
            del combination["batch_size"]
            del combination["epochs"]

            for val_subject in self._y["subject"].unique():
                cur_log_path = join(folder, val_subject)
                makedirs(cur_log_path, exist_ok=True)

                val_mask = self._y["subject"] == val_subject
                X_val = self._X[val_mask]
                y_val = self._y[val_mask]

                X = self._X[~val_mask]
                y = self._y[~val_mask]

                train_subjects = y["subject"].unique()
                train_subjects = train_subjects[:int(0.8 * len(train_subjects))]
                train_mask = y["subject"].isin(train_subjects)

                X_train = X[train_mask]
                y_train = y[train_mask][self._ground_truth].values
                X_test = X[~train_mask]
                y_test = y[~train_mask][self._ground_truth].values

                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
                test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

                model = model_config.model(**combination)
                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    batch_size=batch_size,
                    callbacks=[es],
                    verbose=verbose,
                )

                plt.plot(history.history["loss"], label="train")
                plt.plot(history.history["val_loss"], label="test")
                plt.title(f"Model Loss for {val_subject}")
                plt.legend()
                plt.tight_layout()
                # plt.show()
                plt.savefig(join(cur_log_path, "loss.png"))

                avg_score.add_subject(val_subject, y_val[self._ground_truth].values, model.predict(X_val))
                model.save(join(cur_log_path, "model", "model.h5"))

        df = avg_score.get_final_results()
        df.to_csv(join(folder, "results.csv"))


class SubjectScoresAvg(object):

    def __init__(self):
        self._df = pd.DataFrame()

    def add_subject(self, name, ground_truth, prediction):
        mse = mean_squared_error(ground_truth, prediction)
        mae = mean_absolute_error(ground_truth, prediction)
        mape = mean_absolute_percentage_error(ground_truth, prediction)
        r2 = r2_score(ground_truth, prediction)
        df = pd.DataFrame({"mse": mse, "mae": mae, "mape": mape, "r2": r2}, index=[name])
        self._df = pd.concat([self._df, df], axis=0)

    def get_final_results(self):
        return self._df
