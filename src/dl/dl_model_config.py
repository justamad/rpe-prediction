from src.ml.ml_model_config import LearningModelBase
from src.dl import build_conv_lstm_regression_model
from keras import optimizers
from scikeras.wrappers import KerasRegressor


def build_model():
    model = build_conv_lstm_regression_model(
        n_samples=140,
        n_features=55,
        kernel_size=21,
        n_filters=128,
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mse", "mae", "mape"]
    )
    # model.summary()
    return model


class ConvLSTMModelConfig(LearningModelBase):

    def __init__(self):
        model = KerasRegressor(model=build_model, verbose=0)

        tunable_parameters = {
            f"{str(self)}__optimizer": ["adam"],
            f"{str(self)}__batch_size": [32],
            f"{str(self)}__epochs": [100],
        }

        super().__init__(model=model, grid_search_params=tunable_parameters)

    def __repr__(self):
        return "convlstm"


regression_models = [ConvLSTMModelConfig()]
