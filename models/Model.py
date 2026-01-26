import abc
import datetime
import os.path
import pickle
import numpy as np
import pandas as pd
import itertools
from typing import Dict, Any, Union, Tuple, List
from monipy.data.Scaler import Scaler




class Model(metaclass=abc.ABCMeta):
    """

    """

    def __init__(self, config: Dict[str, Any]):
        """
        Defines interface for framework-agnostic models to be used in train/evaluation scripts
        Args:
            config (dict): the mode parameter configuration
        """

        if "event_filter_id" not in config.keys():
            raise RuntimeError("You need to specify a event_filter_id")

        self.model = None
        self.seizure_types = None
        self.timestamp = None
        self.training_duration = None
        self.database_id = None
        self.metrics = None
        self.provides_thr = False
        self.training_dataset_id = None
        self.valid_dataset_id = None
        self.n_cross_validation_folds = None
        self.exclude_feats_idxs = (
            []
        )  # The list of feature indexes to be excluded from training and validation sets.
        self.event_filter_id = None
        self.config = None
        self.scaler = None
        self.validation_metrics = None

        self.prediction_threshold = 0.5
        self.apply_config(config)
        pass


    def apply_config(self, config: Dict[str, Any]) -> None:
        self.n_cross_validation_folds = config.get("n_cross_validation_folds")
        self.event_filter_id = config["event_filter_id"]
        self.prediction_threshold = config["prediction_threshold"]
        self.scaler = Scaler(config["scaler"])
        self.exclude_feats_idxs = config.get("exclude_feats_idxs", [])
        self.training_dataset_id = config["training_dataset_id"]
        self.valid_dataset_id = config["valid_dataset_id"]

        self.input_shape = (55, 38 - len(self.exclude_feats_idxs))
        self.config = config


    ####################################################################################################################
    # Abstract methods

    @abc.abstractmethod
    def _reset_model(self):
        pass

    @abc.abstractmethod
    def load_internals(self, model_path: str):
        pass

    @abc.abstractmethod
    def predict(self, data: np.ndarray, should_pick_best_iteration: bool = True):
        # The argument should_pick_best_iteration is required for predictions with xgboost (for deep-learning models
        # the best weights are saved at training time). It should be set to False whenever performing cross-validation.
        pass

    @abc.abstractmethod
    def _train_core(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: Union[np.ndarray, None],
        y_valid: Union[np.ndarray, None],
        cv_model_path: str,
        n_fold: int,
        verbose: int = 2,
        grid_search: bool = False,
        should_early_stop: bool = False,
    ):
        pass

    @abc.abstractmethod
    def save(self):
        pass

    ####################################################################################################################
    # Static methods

    @staticmethod
    def load_by_path(path: str):  # TODO: return type should be Model
        with open(os.path.join(path, "monikit_model.pkl"), "rb") as f:
            model = pickle.load(f)

        model.load_internals(path)
        return model

