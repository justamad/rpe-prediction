import types
import tensorflow as tf

from keras.callbacks import EarlyStopping, TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from .win_generator import WinDataGen
from datetime import datetime
from os.path import join


class CustomKerasRegressor(KerasRegressor):

    def __init__(self, build_fn, **kwargs):
        super().__init__(build_fn, **kwargs)
        self.model = None
        self.__history = None

    def fit(self, X, y, **kwargs):
        # taken from keras.wrappers.scikit_learn.KerasClassifier.fit
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        loss_name = self.model.loss
        if hasattr(loss_name, '__name__'):
            loss_name = loss_name.__name__

        # datagen = WinDataGen(X, y, win_size=30, overlap=0.9, batch_size=32, balance=False)

        if "X_val" in kwargs and "y_val" in kwargs:
            X_val = kwargs["X_val"]
            y_val = kwargs["y_val"]
        else:
            val_flow = None
            val_steps = None

        epochs = self.sk_params["epochs"]
        batch_size = self.sk_params["batch_size"]
        print(X.shape, y.shape)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # log_dir = join("logs/fit", timestamp)
        # tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        es_callback = EarlyStopping(monitor="loss", patience=3)
        callbacks = [es_callback]

        train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size=batch_size)
        self.__history = self.model.fit(train_dataset, epochs=epochs, callbacks=callbacks)
        # self.model.save(f"models/{timestamp}/model")
        return self.__history

    # def score(self, X, y, **kwargs):
    #     kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
    #
    #     loss_name = self.model.loss
    #     if hasattr(loss_name, '__name__'):
    #         loss_name = loss_name.__name__
    #
    #     outputs = self.model.evaluate(X, y, **kwargs)
    #     if type(outputs) is not list:
    #         outputs = [outputs]
    #     for name, output in zip(self.model.metrics_names, outputs):
    #         if name == 'acc':
    #             return output
    #     raise Exception('The model is not configured to compute accuracy. '
    #                     'You should pass `metrics=["accuracy"]` to '
    #                     'the `model.compile()` method.')

    @property
    def history(self):
        return self.__history
