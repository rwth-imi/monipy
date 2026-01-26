"""This class implements the Scaler to scale input data for the models"""
import numpy as np
import warnings
from typing import Any
from sklearn.preprocessing import RobustScaler


class Scaler:
    def __init__(self, config) -> None:
        self.config = config
        self.warn = config.get("warn", True)

        # default fit func
        self._fit_func = self._no_fit_at_all
        if config["scaler"] == "scikit_robust_scaler":
            self._fit_func = self._scikit_robust_scaler_fitter
            self._transform_func = self._scikit_robust_scaler_transform
        else:
            raise NotImplementedError(f"{config['scaler']} is not a known scaler")

    def fit(self, data: np.ndarray) -> Any:
        """
        Fit the scaler to the input data

        Args:
            data (np.ndarray): Input data to fit the scaler to.

        Returns:
            Any: Output of the scaler function.

        """

        return self._fit_func(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data via the scaler

        Args:
            data (np.ndarray): Input data.

        Returns:
            ndarray: Transformed input data.

        """
        # data has to be n * 55 * 30
        if not data.shape[1] == 55:
            # todo: back to error when not needed anymore
            if self.warn:
                warnings.warn(f"data has to be n * 55 * m, but has {data.shape}")

        # we dont want wo override the source data
        data2 = np.copy(data)
        return self._transform_func(data2)

    def _no_fit_at_all(self, data: np.ndarray) -> None:
        pass

    def _no_transform_at_all(self, data: np.ndarray) -> np.ndarray:
        return data

    def _scikit_robust_scaler_fitter(self, data: np.ndarray) -> None:
        reshaped_data, orig_shape_data = self._reshape_feats(data)
        self.robust_scaler = RobustScaler().fit(reshaped_data)

    def _scikit_robust_scaler_transform(self, data: np.ndarray) -> np.ndarray:
        reshaped_data, orig_shape_data = self._reshape_feats(data)
        data_scaled = self._reshape_feats_inv(
            self.robust_scaler.transform(reshaped_data), orig_shape_data
        )
        return data_scaled