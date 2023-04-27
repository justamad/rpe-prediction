from .models import build_conv_model, build_cnn_lstm_model
from src.ml.ml_model_config import LearningModelBase, parse_report_file_to_model_parameters
from scikeras.wrappers import KerasRegressor

import pandas as pd


class ConvModelConfig(LearningModelBase):

    def __init__(self):
        model = KerasRegressor(
            model=build_conv_model,
            n_layers=3, n_filters=32, kernel_size=(10,3), dropout=0.3, n_units=128,
            verbose=False,
        )

        tunable_parameters = {
            f"{str(self)}__batch_size": [64],
            f"{str(self)}__epochs": [200],
            f"{str(self)}__n_filters": [16],
            f"{str(self)}__n_layers": [2, 3],
            f"{str(self)}__kernel_size": [(10, 3)],
            f"{str(self)}__dropout": [0.3],
            f"{str(self)}__n_units": [128, 256],
        }

        super().__init__(model=model, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "conv"


class CNNLSTMModelConfig(LearningModelBase):

    def __init__(self):
        model = KerasRegressor(
            model=build_cnn_lstm_model,
            n_filters=32, kernel_size=(10, 3), n_layers=3, dropout=0.3, lstm_units=50,
            verbose=False,
        )

        tunable_parameters = {
            f"{str(self)}__batch_size": [64],
            f"{str(self)}__epochs": [200],
            f"{str(self)}__n_filters": [16],
            f"{str(self)}__n_layers": [2, 3],
            f"{str(self)}__kernel_size": [(10, 3)],
            f"{str(self)}__dropout": [0.3],
            f"{str(self)}__lstm_units": [8, 16],
        }

        super().__init__(model=model, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "cnnlstm"


regression_models = [ConvModelConfig(), CNNLSTMModelConfig()]
models = {str(model): model for model in regression_models}


def instantiate_best_dl_model(result_df: pd.DataFrame, model_name: str, task: str):
    if model_name not in models:
        raise AttributeError(f"Model {model_name} not found.")

    if task == "regression":
        column = "rank_test_r2"
    elif task == "classification":
        column = "rank_test_accuracy"
    else:
        raise ValueError(f"Task {task} not supported.")

    best_configuration = parse_report_file_to_model_parameters(result_df, model_name, column)
    best_configuration["verbose"] = 1
    # best_configuration["epochs"] = 1
    model = models[model_name].model
    model.set_params(**best_configuration)
    return model
