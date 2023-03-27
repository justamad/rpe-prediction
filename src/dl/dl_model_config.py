from src.ml.ml_model_config import LearningModelBase, parse_report_file_to_model_parameters
from src.dl import build_conv_lstm_regression_model
from scikeras.wrappers import KerasRegressor
from typing import List
# from keras import optimizers
from tensorflow import keras  # resolve bug with Tensorflow 2.11.0 and appearantly 2.12.0

import pandas as pd


def build_model(n_samples: int, n_features: int):
    model = build_conv_lstm_regression_model(
        n_samples=n_samples,
        n_features=n_features,
        kernel_size=21,
        n_filters=128,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse", "mae", "mape"]
    )
    # model.summary()
    return model


class ConvLSTMModelConfig(LearningModelBase):

    def __init__(self, n_samples, n_features):
        model = KerasRegressor(model=build_model(n_samples, n_features), optimizer="adam", epochs=30, batch_size=32)

        tunable_parameters = {
            # f"{str(self)}__optimizer": ["adam"],
            f"{str(self)}__batch_size": [16, 32],
            f"{str(self)}__epochs": [20],
        }

        super().__init__(model=model, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "convlstm"


regression_models = [ConvLSTMModelConfig]
models = {
    "convlstm": build_model,
}


def build_regression_models(n_samples: int, n_features: int) -> List[LearningModelBase]:
    config_models = []
    for model in regression_models:
        config_models.append(model(n_samples, n_features))
    return config_models


def instantiate_best_dl_model(result_df: pd.DataFrame, model_name: str, metric: str, n_samples: int, n_features: int):
    if model_name not in models:
        raise AttributeError(f"Model {model_name} not found.")

    best_configuration = parse_report_file_to_model_parameters(result_df, metric, model_name)
    model = models[model_name](n_samples, n_features)
    # model = build_model(n_samples=best_model["n_samples"], n_features=best_model["n_features"])
    # model.set_params(**best_model["params"])
    return model, best_configuration
