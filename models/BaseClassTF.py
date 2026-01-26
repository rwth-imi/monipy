import os

import abc
import numpy as np
import os
import pickle
from monipy.models.Model import Model

#import tensorflow as tf
import numpy as np
import os
import random

# Set a seed value
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value




class BaseClassTF(Model):
    def __init__(self, config):
        super().__init__(config)

    ####################################################################################################################
    # inherited methods

    @abc.abstractmethod
    def _reset_model(self):
        pass

    @abc.abstractmethod
    def load_internals(self, model_path: str):
        pass

    def predict(
        self, data: np.ndarray, should_pick_best_iteration: bool = True
    ) -> np.array:
        """
        The function to be called to run inference Args: data (numpy.array): the input data (features)
        should_pick_best_iteration (bool): in this context it is a dummy argument, that is required by xgboost only.
        However, it needs to be included for consistency with the parent Model class.

        Returns (numpy.array): model detections

        """

        if data.shape[0] == 0:
            return None
        return self.model.predict(self.scaler.transform(data)).flatten()

    def _train_core(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        verbose: int = 0,
    ) -> None:
        """
        The core of the fitting process, including setting the optimizer and callbacks
        Args:
            x_train (numpy.array): the features of the training set on the cv fold
            y_train (numpy.array): the labels of the training set on the cv fold
            x_valid (numpy.array): the features of the validation set on the cv fold
            y_valid (numpy.array): the labels of the training set on the cv fold
            verbose (str): the level of verbose (see keras)
            grid_search (bool): whether the fit occurs inside a grid search

        Returns (None):

        """
        historys = list()
        x_train = self.reshape_data(x_train)
        x_valid = self.reshape_data(x_valid)

        cls_weight = self.config["class_weight"]

        train_params = {
            "batch_size": self.config["batch_size"],
            "epochs": self.config["epochs"],
            "shuffle": self.config["shuffle"],
            "class_weight": {0: cls_weight[0], 1: cls_weight[1]},
        }

        cbks = None

        if x_valid is not None:

            history = self.model.fit(
                x_train,
                y_train.astype(np.float32),
                validation_data=(x_valid, y_valid.astype(np.float32)),
                validation_split = 0.5,
                callbacks=cbks,
                verbose=verbose,
                **train_params,
            )
            historys.append(history)

        else:
            history = self.model.fit(
                x_train,
                y_train.astype(np.float32),
                callbacks=cbks,
                verbose=verbose,
                **train_params,
            )
            historys.append(history)


        return historys

    def _clear_search_session(
        self,
    ):
        pass

    def save(self) -> None:
        """
        Saves the model as a pickle object
        Returns (pickle file): saved model

        """
        self.model.save(self.get_model_path())
        model_temp = self.model
        self.model = None
        with open(os.path.join(self.get_model_path(), "monikit_model.pkl"), "wb") as f:
            pickle.dump(self, f)
        self.model = model_temp

