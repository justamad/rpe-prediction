import logging
import os
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import yaml

from .plot_callback import PerformancePlotCallback
from .win_generator import WinDataGen
from contextlib import redirect_stdout
from src.ml import MLOptimization, LearningModelBase
from typing import Union, List, Dict
from sklearn.model_selection import ParameterGrid
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
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        base_folder = join(log_path, str(model_config))
        grid = ParameterGrid(model_config.parameters)

        total_fits = len(grid) * self._n_splits
        print(f"Grid search with {len(grid)} combinations for {self._n_splits} folds. Total {total_fits} fits.")

        for combination_idx, combination in enumerate(grid):
            cur_folder = join(base_folder, f"combination_{combination_idx}")
            os.makedirs(cur_folder, exist_ok=True)
            yaml.dump(combination, open(join(cur_folder, "params.yaml"), "w"))

            intermediate_result_df = pd.DataFrame()

            # Extract meta parameters for training
            batch_size = combination["batch_size"]
            epochs = combination["epochs"]
            win_size = combination["win_size"]
            overlap = combination["overlap"]
            for key in ["batch_size", "epochs", "overlap"]:
                del combination[key]

            for sub_idx, val_subject in enumerate(self._subjects):
                logging.info(f"Start [{combination_idx}/{len(grid) - 1}] - [{sub_idx}/{len(self._subjects) - 1}]")
                subjects = list(self._subjects)
                subjects.remove(val_subject)
                test_subjects = random.sample(subjects, 1)
                train_subjects = [s for s in subjects if s not in test_subjects]

                X_val, y_val = self._X[self._y["subject"] == val_subject], self._y[self._y["subject"] == val_subject]
                X_test, y_test = self._X[self._y["subject"].isin(test_subjects)], self._y[self._y["subject"].isin(test_subjects)]
                X_train, y_train = self._X[self._y["subject"].isin(train_subjects)], self._y[self._y["subject"].isin(train_subjects)]

                if lstm:
                    train_dataset = WinDataGen(X_train, y_train, self._ground_truth, win_size, overlap, batch_size, shuffle=True, balance=True)
                    train_view_dataset = WinDataGen(X_train, y_train, self._ground_truth, win_size, overlap, batch_size, shuffle=False, balance=False)
                    test_dataset = WinDataGen(X_test, y_test, self._ground_truth, win_size, overlap, batch_size, shuffle=False, balance=False)
                    val_dataset = WinDataGen(X_val, y_val, self._ground_truth, win_size, overlap, batch_size, shuffle=False, balance=False, deliver_sets=True)
                else:
                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
                    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
                    train_view_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

                plot_cb = PerformancePlotCallback(train_view_dataset, test_dataset, val_dataset, join(cur_folder, val_subject))
                model = model_config.model(**combination)

                with open(join(cur_folder, 'modelsummary.txt'), 'w') as f:
                    with redirect_stdout(f):
                        model.summary()

                history = model.fit(
                    train_dataset,
                    epochs=epochs,
                    validation_data=test_dataset,
                    batch_size=batch_size,
                    callbacks=[es, plot_cb],
                )

                predictions, ground_truth, sets = [], [], []
                for i in range(len(val_dataset)):
                    X, y = val_dataset[i]
                    if len(y.shape) == 2:
                        sets.extend(y[:, 1].reshape(-1))
                        y = y[:, 0]
                    else:
                        sets.extend([0 for _ in range(len(y))])

                    predictions.extend(model.predict(X).reshape(-1))
                    ground_truth.extend(y.reshape(-1))

                df = pd.DataFrame({"prediction": predictions, "ground_truth": ground_truth, "set_id": sets})
                df["subject"] = val_subject
                intermediate_result_df = pd.concat([intermediate_result_df, df], axis=0)

                plt.plot(history.history["loss"], label="train")
                plt.plot(history.history["val_loss"], label="test")
                plt.title(f"Model Loss for {val_subject}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(join(cur_folder, f"{val_subject}_loss.png"))
                plt.close()
                plt.clf()

            intermediate_result_df.to_csv(join(cur_folder, "results.csv"))
