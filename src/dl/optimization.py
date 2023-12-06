import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from .plot_callback import ProgressPlotCallback
from .models import build_cnn_lstm_model, build_cnn_gru_model, build_cnn_fc_model
from .seq_generator import SequenceGenerator
from typing import Union
from keras.callbacks import EarlyStopping
from kerastuner.tuners import BayesianOptimization
from os.path import join, exists


class DLOptimization(object):

    def __init__(self, X: Union[pd.DataFrame, np.ndarray], y: pd.DataFrame, **kwargs):
        super().__init__()
        self._X = X
        self._y = y
        self._subjects = sorted(self._y["subject"].unique())

        self._balance = kwargs["balance"]
        self._ground_truth = kwargs["ground_truth"]
        self._task = kwargs["task"]
        self._mode = kwargs["mode"]
        self._epochs = kwargs["epochs"]
        self._batch_size = kwargs["batch_size"]
        self._win_size = kwargs["win_size"]
        self._overlap = kwargs["overlap"]
        self._patience = kwargs["patience"]
        self._verbose = kwargs["verbose"]
        self._max_iter = kwargs["max_iter"]
        self._test_subjects = kwargs["test_subjects"]
        self._val_subjects = kwargs["val_subjects"]

    def perform_grid_search_with_cv(self, log_path: str):
        models = {"CNN-FC": build_cnn_fc_model, "CNN-GRU": build_cnn_gru_model, "CNN-LSTM": build_cnn_lstm_model}
        for model in models:
            print(f"Start grid search for model: {model}")
            self.train_model(models[model], join(log_path, model))

    def train_model(self, build_fn, log_path: str):
        es = EarlyStopping(monitor="val_mse", patience=self._patience, restore_best_weights=True, start_from_epoch=10)
        n_features = self._X[0].shape[-1]

        if len(self._subjects) % self._val_subjects != 0:
            print(f"WARNING: Number of subjects ({len(self._subjects)}) is not divisible by ({self._val_subjects})")

        n_folds = math.ceil(len(self._subjects) / self._val_subjects)
        np.random.seed(42)
        for fold_id in range(n_folds):
            print(f"Start fold: [{fold_id + 1}/{n_folds}]")
            cur_log_path = join(log_path, f"Fold_{fold_id:02d}")

            if exists(join(cur_log_path, "eval_dataset.csv")):
                print(f"Fold {fold_id} already exists. Skipping...")
                continue

            cur_idx = fold_id * self._val_subjects
            validation_subjects = self._subjects[cur_idx:cur_idx + self._val_subjects]
            subjects = [subject for subject in self._subjects if subject not in validation_subjects]
            test_subjects = random.sample(subjects, self._test_subjects)
            train_subjects = [subject for subject in subjects if subject not in test_subjects]

            train_mask = self._y["subject"].isin(train_subjects)
            test_mask = self._y["subject"].isin(test_subjects)
            val_mask = self._y["subject"].isin(validation_subjects)

            X_val, y_val = self._X[val_mask], self._y[val_mask]
            X_test, y_test = self._X[test_mask], self._y[test_mask]
            X_train, y_train = self._X[train_mask], self._y[train_mask]

            train_gen = SequenceGenerator(
                X_train, y_train, self._ground_truth, self._win_size, self._overlap, self._batch_size, shuffle=True,
                balance=self._balance,
            )

            test_gen = SequenceGenerator(
                X_test, y_test, self._ground_truth, self._win_size, self._overlap, self._batch_size,
                shuffle=False, balance=False
            )

            tuner = BayesianOptimization(
                lambda hp: build_fn(hp, self._win_size, n_features=n_features),
                objective='val_mse',
                max_trials=self._max_iter,
                directory=cur_log_path,
                project_name="optimization",
            )
            tuner.search(train_gen, epochs=self._epochs, validation_data=test_gen, verbose=self._verbose,
                         callbacks=[es])
            result_df = save_trials_to_dataframe(tuner)
            result_df.to_csv(join(cur_log_path, f"tune_results.csv"))

            train_val_gen = SequenceGenerator(
                X_train, y_train, self._ground_truth, self._win_size, self._overlap, self._batch_size,
                shuffle=False, balance=False,
            )

            val_gen = SequenceGenerator(
                X_val, y_val, self._ground_truth, self._win_size, self._overlap, self._batch_size,
                shuffle=False, balance=False, meta_data=True,
            )

            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            model = tuner.hypermodel.build(best_hps)
            plot_cb = ProgressPlotCallback(train_val_gen, test_gen, val_gen, join(cur_log_path, "losses"), gen_step=1)
            history = model.fit(
                train_gen,
                epochs=self._epochs,
                validation_data=test_gen,
                verbose=self._verbose,
                callbacks=[es, plot_cb],
            )

            predictions = []
            labels = []
            for i in range(len(val_gen)):
                X_batch, y_batch = val_gen[i]
                pred = np.array(model(X_batch, training=False))
                predictions.extend(list(pred.reshape(-1)))
                labels.extend(list(y_batch))

            labels = np.array(labels)
            eval_dataset = pd.DataFrame({
                "prediction": predictions,
                "ground_truth": labels[:, 0],
                "set_id": labels[:, 1],
                "subject": labels[:, 2],
            })
            eval_dataset.to_csv(join(cur_log_path, "eval_dataset.csv"))

            with open(join(cur_log_path, "model_summary.txt"), "w") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))

            plt.plot(history.history["loss"], label="train")
            plt.plot(history.history["val_loss"], label="validation")
            plt.title("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(join(cur_log_path, f"loss.png"))
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
