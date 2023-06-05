import pandas as pd

from .models import build_conv2d_model, build_cnn_lstm_model
from src.ml.ml_model_config import LearningModelBase, parse_report_file_to_model_parameters
# from scikeras.wrappers import KerasRegressor


class ConvModelConfig(LearningModelBase):

    def __init__(self):
        regressor = build_conv2d_model

        tunable_parameters = {
            f"batch_size": [16, 32],
            f"learning_rate": [1e-4],
            f"epochs": [500],
            f"n_filters": [32, 64, 128],
            f"n_layers": [1, 2, 3],
            f"kernel_size": [(3, 3)],
            f"dropout": [0.5],
            f"n_units": [128],
        }

        super().__init__(model=regressor, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "Conv2D"


class CNNLSTMModelConfig(LearningModelBase):

    def __init__(self):
        model = build_cnn_lstm_model
        #     KerasRegressor(
        #     model=build_cnn_lstm_model,
        #     n_filters=32, kernel_size=(10, 3), n_layers=3, dropout=0.3, lstm_units=50,
        #     verbose=False,
        # )

        tunable_parameters = {
            "n_filters": [64, 128],
            "n_layers": [2, 3],
            "kernel_size": [(3, 3), (5, 3)],
            "dropout": [0.5],
            "lstm_units": [64, 128],
            "batch_size": [16],
            "epochs": [500],
            "win_size": [30, 60, 90, 120],
            "overlap": [0.9],
        }

        super().__init__(model=model, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "CNNLSTM"


# regression_models = [ConvModelConfig()]


def instantiate_best_dl_model(result_df: pd.DataFrame, model_name: str, task: str):
    models = {str(model): model for model in [CNNLSTMModelConfig(), ConvModelConfig()]}

    if task == "regression":
        column = "avg_mse"
    else:
        raise ValueError(f"Task {task} not supported.")

    best_params_dict = parse_report_file_to_model_parameters(result_df, model_name, column)
    best_params_dict = {k.replace("param_", ""): v for k, v in best_params_dict.items()}

    meta_dict = {}
    for reserved_key in ["epochs", "overlap", "batch_size"]:
        meta_dict[reserved_key] = best_params_dict.pop(reserved_key)

    if "win_size" in best_params_dict:
        meta_dict["win_size"] = best_params_dict["win_size"]  # TODO: This is hacky...

    model = models[model_name].model(**best_params_dict)
    return model, meta_dict
