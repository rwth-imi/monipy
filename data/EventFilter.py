import pandas as pd
import numpy as np

from typing import Dict, Any


class EventFilter:
    def __init__(self, feature_data: pd.DataFrame, funcs: Dict[str, Any]):
        """

        Args:
            feature_data (): 30 sec features, shape 55 x 13
            funcs ():
        """
        self.feature_data = feature_data

        self.filter_functions = {
            "filter_longterm_tachycardy": self.filter_longterm_tachycardy,
        }

        self.is_event = np.all(
            [self.filter_functions[name](funcs[name]) for name in funcs]
        )


    def filter_longterm_tachycardy(self, params: Dict[str, Any]) -> bool:
        heartrate = (60000 / self.feature_data.avg).to_numpy()

        n_tachycardy = len(heartrate[heartrate > params["hr_threshold"]])
        frac_tachycardy = n_tachycardy / len(heartrate)

        return frac_tachycardy > params["tachycardy_fraction"]
