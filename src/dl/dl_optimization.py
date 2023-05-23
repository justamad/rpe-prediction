import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.ml import MLOptimization, LearningModelBase
from typing import Union, List, Dict
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
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = join(log_path, str(model_config), timestamp)
        grid = ParameterGrid(model_config.parameters)
        print(f"Starting grid search with {len(grid)} combinations for {self._n_splits} folds. Total {len(grid) * self._n_splits} fits.")

        avg_score = SubjectScoresAvg()
        for combination_idx, combination in enumerate(grid):
            avg_score.set_new_combination(combination)
            cur_log_path = join(folder, str(combination_idx))
            makedirs(cur_log_path, exist_ok=True)

            batch_size = combination["batch_size"]
            epochs = combination["epochs"]
            del combination["batch_size"]
            del combination["epochs"]

            validation_subjects = self._y["subject"].unique()
            for sub_idx, val_subject in enumerate(validation_subjects):
                print(f"Start [{combination_idx}/{len(grid) - 1}] - [{sub_idx}/{len(validation_subjects) - 1}]")

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
                    verbose=0,
                )

                plt.plot(history.history["loss"], label="train")
                plt.plot(history.history["val_loss"], label="test")
                plt.title(f"Model Loss for {val_subject}")
                plt.legend()
                plt.tight_layout()
                # plt.show()
                plt.savefig(join(cur_log_path, f"{val_subject}_loss.png"))
                plt.close()
                plt.clf()

                fig, axs = plt.subplots(3, 1)
                axs[0].set_title("Train")
                axs[0].plot(model.predict(X_train, verbose=0), label="Prediction")
                axs[0].plot(y_train, label="Ground Truth")
                axs[1].set_title("Test")
                axs[1].plot(model.predict(X_test, verbose=0), label="Prediction")
                axs[1].plot(y_test, label="Ground Truth")
                axs[2].set_title("Validation")
                axs[2].plot(model.predict(X_val, verbose=0), label="Prediction")
                axs[2].plot(y_val[self._ground_truth].values, label="Ground Truth")
                plt.legend()
                plt.tight_layout()
                plt.savefig(join(cur_log_path, f"{val_subject}_results.png"))
                plt.close()
                plt.clf()

                avg_score.add_subject_results(val_subject, y_val[self._ground_truth].values, model.predict(X_val))
                # model.save(join(cur_log_path, "model", "model.h5"))

        df = avg_score.get_final_results()
        df.to_csv(join(folder, "results.csv"))


class SubjectScoresAvg(object):

    def __init__(self):
        self._df = pd.DataFrame()
        self._cur_row = None

    def set_new_combination(self, param_combination: Dict):
        if self._cur_row is not None:
            self._append_row()

        self._cur_row = {f"param_{k}": v for k, v in param_combination.items()}

    def _append_row(self):
        for metric in ["mse", "mae", "mape", "r2"]:
            self._cur_row[f"avg_{metric}"] = np.mean([v for k, v in self._cur_row.items() if metric in k])
            self._cur_row[f"std_{metric}"] = np.std([v for k, v in self._cur_row.items() if metric in k])

        df = pd.DataFrame(data={k: str(v) if isinstance(v, tuple) else v for k, v in self._cur_row.items()}, index=[0])
        self._df = pd.concat([self._df, df], axis=0, ignore_index=True)

    def add_subject_results(self, name: str, ground_truth: np.ndarray, prediction: np.ndarray):
        mse = mean_squared_error(ground_truth, prediction)
        mae = mean_absolute_error(ground_truth, prediction)
        mape = mean_absolute_percentage_error(ground_truth, prediction)
        r2 = r2_score(ground_truth, prediction)
        for metric, value in zip(["mse", "mae", "mape", "r2"], [mse, mae, mape, r2]):
            self._cur_row[f"{name}_{metric}"] = value

    def get_final_results(self):
        self._append_row()
        return self._df
