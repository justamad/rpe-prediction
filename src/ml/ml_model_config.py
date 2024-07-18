from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from typing import Dict, Union, Tuple

import pandas as pd
import ast
import math


class LearningModelBase(object):

    def __init__(self, model, grid_search_params: Dict):
        self.__model = model
        self.__grid_search_params = grid_search_params

    @property
    def model(self):
        return self.__model

    @property
    def parameters(self):
        return self.__grid_search_params


class SVRModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {
            f"{str(self)}__kernel": ["rbf"],  # , "linear"],
            f"{str(self)}__gamma": [1e-3, 1e-4],
            f"{str(self)}__C": [1e0, 1e1, 1e2, 1e3],
        }

        super().__init__(model=SVR(), grid_search_params=tuned_parameters)

    def __repr__(self):
        return "svr"


class RFModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {
            f"{str(self)}__n_estimators": [50, 100, 150, 200],
            f"{str(self)}__criterion": ["squared_error"]  # "absolute_error"
        }

        model = RandomForestRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "rf"


class GBRModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {
            f"{str(self)}__n_estimators": [150, 500],
            f"{str(self)}__learning_rate": [0.05, 0.1, 0.2],
            f"{str(self)}__loss": ["squared_error"],  # , "absolute_error", "huber"],
            f"{str(self)}__n_iter_no_change": [None, 5, 50]
        }

        model = GradientBoostingRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "gbr"


class MLPModelConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {
            f'{str(self)}__hidden_layer_sizes': [(100,), (100, 50), (100, 150)],
            f'{str(self)}__activation': ["logistic", "relu", "tanh"],
            f'{str(self)}__solver': ["adam"],
            f'{str(self)}__learning_rate_init': [1e-3, 1e-4],
            f'{str(self)}__max_iter': [5000],
            f'{str(self)}__early_stopping': [True],
        }

        model = MLPRegressor(batch_size=32)
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "mlp"


class XGBoostConfig(LearningModelBase):

    def __init__(self):
        tuned_parameters = {
            f"{str(self)}__n_estimators": [400, 700, 1000],
            f"{str(self)}__colsample_bytree": [0.7, 0.8],
            f"{str(self)}__max_depth": [15, 20, 25],
            f"{str(self)}__reg_alpha": [1.1, 1.2, 1.3],
            f"{str(self)}__reg_lambda": [1.1, 1.2, 1.3],
            f"{str(self)}__subsample": [0.7, 0.8, 0.9]
        }
        model = XGBRegressor()
        super().__init__(model=model, grid_search_params=tuned_parameters)

    def __repr__(self):
        return "XGBoost"


regression_models = [SVRModelConfig(), RFModelConfig(), GBRModelConfig(), MLPModelConfig()]
models = {str(model): model for model in regression_models}


def instantiate_best_model(result_df: pd.DataFrame, model_name: str, task: str):
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found.")

    if task == "regression":
        column = "rank_test_r2"
    elif task == "classification":
        column = "rank_test_accuracy"
    else:
        raise ValueError(f"Task {task} not supported.")

    best_parameters = parse_report_file_to_model_parameters(result_df, model_name, column)
    return models[model_name].model.set_params(**best_parameters)


def parse_report_file_to_model_parameters(result_df: pd.DataFrame, model_name: str, column: str) -> Dict[str, float]:
    best_combination = result_df.sort_values(by=column, ascending=True).iloc[0]
    best_combination = best_combination[best_combination.index.str.contains("param")]
    params = {k.replace(f"param_{model_name}__", ""): parse_types(v) for k, v in best_combination.to_dict().items()}
    return params


def parse_types(value) -> Union[int, float, str, Tuple, None]:
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        if math.isnan(value):
            return None

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass

    return value
