import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from .plot_callback import PerformancePlotCallback
from .win_generator import WinDataGen
from src.ml import MLOptimization, LearningModelBase
from typing import Union, List, Dict
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
from os import makedirs
from os.path import join


class DLOptimization(MLOptimization):

    def __init__(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: pd.DataFrame,
            balance: bool,
            task: str,
            mode: str,
            ground_truth: Union[str, List[str]],
            n_splits: int = None
    ):
        super().__init__(X, y, balance, ground_truth, task, mode, n_splits)
        self._subjects = self._y["subject"].unique()

    def perform_grid_search_with_cv(
            self,
            model_config: LearningModelBase,
            log_path: str,
            view_progress: int = 1,
            verbose: int = 1,
            patience: int = 3,
            lstm: bool = True,
    ):
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder = join(log_path, str(model_config), timestamp)
        grid = ParameterGrid(model_config.parameters)

        total_fits = len(grid) * self._n_splits
        print(f"Grid search with {len(grid)} combinations for {self._n_splits} folds. Total {total_fits} fits.")

        avg_score = SubjectScoresAvg()
        for combination_idx, combination in enumerate(grid):
            avg_score.set_new_combination(combination)

            batch_size = combination["batch_size"]
            epochs = combination["epochs"]
            win_size = combination["win_size"]
            overlap = combination["overlap"]
            for key in ["batch_size", "epochs", "win_size", "overlap"]:
                del combination[key]

            for sub_idx, val_subject in enumerate(self._subjects):
                print(f"Start [{combination_idx}/{len(grid) - 1}] - [{sub_idx}/{len(self._subjects) - 1}]")
                subjects = list(self._subjects)
                subjects.remove(val_subject)
                test_subjects = random.sample(subjects, 3)
                train_subjects = [s for s in subjects if s not in test_subjects]

                X_val, y_val = self._X[self._y["subject"] == val_subject], self._y[self._y["subject"] == val_subject]
                X_test, y_test = self._X[self._y["subject"].isin(test_subjects)], self._y[self._y["subject"].isin(test_subjects)]
                X_train, y_train = self._X[self._y["subject"].isin(train_subjects)], self._y[self._y["subject"].isin(train_subjects)]

                if lstm:
                    train_dataset = WinDataGen(X_train, y_train[self._ground_truth].values, win_size, overlap, batch_size, shuffle=True, balance=True)
                    test_dataset = WinDataGen(X_test, y_test[self._ground_truth].values, win_size, overlap, batch_size, shuffle=False, balance=False)
                    train_view_dataset = WinDataGen(X_train, y_train[self._ground_truth].values, win_size, overlap, batch_size, shuffle=False, balance=False)
                    val_dataset = WinDataGen(X_val, y_val[self._ground_truth].values, win_size, overlap, 2, shuffle=False, balance=False)
                else:
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
                    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
                    train_view_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

                plot_cb = PerformancePlotCallback(train_view_dataset, test_dataset, val_dataset, join(log_path, f"combi_{combination_idx}", val_subject))
                model = model_config.model(**combination)
                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    batch_size=batch_size,
                    callbacks=[es, plot_cb],
                )

                plt.plot(history.history["loss"], label="train")
                plt.plot(history.history["val_loss"], label="test")
                plt.title(f"Model Loss for {val_subject}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(join(log_path, f"combi_{combination_idx}", f"{val_subject}_loss.png"))
                plt.close()
                plt.clf()

                avg_score.add_subject_results(val_subject, val_dataset, model)
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

    def add_subject_results(self, name: str, val_generator, model):
        prediction, ground_truth = [], []
        for X, y in val_generator:
            prediction.extend(model.predict(X).reshape(-1))
            ground_truth.extend(y.reshape(-1))

        mse = mean_squared_error(ground_truth, prediction)
        mae = mean_absolute_error(ground_truth, prediction)
        mape = mean_absolute_percentage_error(ground_truth, prediction)
        r2 = r2_score(ground_truth, prediction)
        for metric, value in zip(["mse", "mae", "mape", "r2"], [mse, mae, mape, r2]):
            self._cur_row[f"{name}_{metric}"] = value

    def get_final_results(self):
        self._append_row()
        return self._df
