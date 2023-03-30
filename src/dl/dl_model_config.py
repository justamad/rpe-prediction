from src.ml.ml_model_config import LearningModelBase, parse_report_file_to_model_parameters
from src.dl import build_conv1d_lstm_regression_model
from scikeras.wrappers import KerasRegressor

import pandas as pd


class ConvLSTMModelConfig(LearningModelBase):

    def __init__(self):
        model = KerasRegressor(model=build_conv1d_lstm_regression_model, n_filters=32, kernel_size=10)

        tunable_parameters = {
            f"{str(self)}__batch_size": [64],
            f"{str(self)}__epochs": [500],
            f"{str(self)}__n_filters": [16],
            f"{str(self)}__kernel_size": [9],
        }

        super().__init__(model=model, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "convlstm"


regression_models = [ConvLSTMModelConfig()]
models = {str(model) for model in regression_models}


def instantiate_best_dl_model(result_df: pd.DataFrame, model_name: str, task: str, n_samples: int, n_features: int):
    if model_name not in models:
        raise AttributeError(f"Model {model_name} not found.")

    if task == "regression":
        column = "rank_test_r2"
    elif task == "classification":
        column = "rank_test_accuracy"
    else:
        raise ValueError(f"Task {task} not supported.")

    best_configuration = parse_report_file_to_model_parameters(result_df, model_name, column)
    model = models[model_name]
    # model = build_model(n_samples=best_model["n_samples"], n_features=best_model["n_features"])
    # model.set_params(**best_model["params"])
    return model, best_configuration
