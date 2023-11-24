import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from .plot_callback import PerformancePlotCallback
from .models import build_cnn_lstm_model
from .seq_generator import SequenceGenerator
from kerastuner.tuners import BayesianOptimization
from typing import Union, List
from os.path import join


class DLOptimization(object):

    def __init__(
            self,
            X: Union[pd.DataFrame, np.ndarray],
            y: pd.DataFrame,
            balance: bool,
            task: str,
            mode: str,
            ground_truth: Union[str, List[str]],
            n_splits: int = None,
    ):
        super().__init__()
        self._X = X
        self._y = y
        self._balance = balance
        self._ground_truth = ground_truth
        self._task = task
        self._mode = mode
        self._n_splits = n_splits
        self._subjects = self._y["subject"].unique()
        self._test_subjects = 3

    def perform_grid_search_with_cv(
            self,
            log_path: str,
            epochs: int,
            batch_size: int,
            win_size: int,
            overlap: float,
            patience: int,
            verbose: int,
            max_iter: int,
    ):
        es = tf.keras.callbacks.EarlyStopping(monitor="val_mse", patience=patience, restore_best_weights=True)
        n_features = self._X[0].shape[-1]

        for sub_idx, val_subject in enumerate(self._subjects):
            print(f"Start [{sub_idx}/{len(self._subjects) - 1}] - [{val_subject}]")
            cur_log_path = join(log_path, val_subject)

            subjects = list(self._subjects)
            subjects.remove(val_subject)
            test_subjects = random.sample(subjects, self._test_subjects)
            test_mask = self._y["subject"].isin(test_subjects)

            X_val, y_val = self._X[self._y["subject"] == val_subject], self._y[self._y["subject"] == val_subject]
            X_test, y_test = self._X[test_mask], self._y[test_mask]
            X_train, y_train = self._X[~test_mask], self._y[~test_mask]

            train_dataset = SequenceGenerator(
                X_train, y_train, self._ground_truth, win_size, overlap, batch_size, shuffle=True,
                balance=self._balance,
            )

            train_view_dataset = SequenceGenerator(
                X_train, y_train, self._ground_truth, win_size, overlap, batch_size,
                shuffle=False, balance=False,
            )

            test_dataset = SequenceGenerator(
                X_test, y_test, self._ground_truth, win_size, overlap, batch_size,
                shuffle=False, balance=False
            )

            val_dataset = SequenceGenerator(
                X_val, y_val, self._ground_truth, win_size, overlap, batch_size,
                shuffle=False, balance=False, deliver_sets=False,
            )

            tuner = BayesianOptimization(
                lambda hp: build_cnn_lstm_model(hp, win_size, n_features=n_features),
                objective='val_mse',
                max_trials=max_iter,
                directory=cur_log_path,
                project_name="CNN-LSTM",
            )

            plot_cb = PerformancePlotCallback(
                train_view_dataset, test_dataset, val_dataset, join(cur_log_path, val_subject), gen_step=1,
            )

            tuner.search(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=verbose, callbacks=[es])
            result_df = save_trials_to_dataframe(tuner)
            result_df.to_csv(join(cur_log_path, f"{val_subject}_tune_res.csv"))

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)
            history = model.fit(
                train_dataset,
                epochs=epochs,
                validation_data=test_dataset,
                verbose=verbose,
                callbacks=[es, plot_cb],
            )

            # Write model summary to file
            with open(join(cur_log_path, f"{val_subject}_model_summary.txt"), "w") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))

            plt.plot(history.history["loss"], label="train")
            plt.plot(history.history["val_loss"], label="test")
            plt.title(f"Model Loss for {val_subject}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(join(cur_log_path, f"{val_subject}_loss.png"))
            plt.close()
            plt.clf()


def save_trials_to_dataframe(tuner: BayesianOptimization) -> pd.DataFrame:
    grid_search_df = pd.DataFrame()
    for trial in tuner.oracle.trials:
        trial_state = tuner.oracle.trials[trial].get_state()
        trial_hyperparameters = pd.Series(
            trial_state["hyperparameters"]["values"],
            index=trial_state["hyperparameters"]["values"].keys()
        )
        trial_loss = pd.Series(trial_state["score"], index=["val_loss"])
        trial_tune_res = pd.concat([trial_hyperparameters, trial_loss])
        trial_tune_res.name = trial
        grid_search_df = pd.concat([grid_search_df, trial_tune_res], axis=1)

    grid_search_df = grid_search_df.T
    return grid_search_df
