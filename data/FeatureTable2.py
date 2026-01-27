import numpy as np
import pandas as pd
from typing import Tuple
import monipy.utils.utils as utils


class FeatureTable:
    """
    Interface-compatible FeatureTable implementation.

    Provides:
    - features_30
    - features_60
    - features_all
    - filter_table

    with shapes and public API matching the original FeatureTable2.
    """

    SLIDING_WINDOW_STEP_SIZE = 5            # seconds
    ALL_FEATURES_WINDOW_WIDTH = 300         # seconds

    FEATURE_NAMES_30 = [
        "avg","sd","rmssd","rmssd_dt","skew","kurt","pnnx","nnx",
        "triangular_index","quantile_25","quantile_50","quantile_75","variance"
    ]

    FEATURE_NAMES_60 = [
        "csi","csim","cvi","sd1","sd2",
        "csi_slope","csim_slope","csi_filtered","csim_filtered",
        "csi_filtered_slope","csim_filtered_slope",
        "hr_diff","hr_diff_slope","hr_diff_filtered","hr_diff_filtered_slope"
    ]

    def __init__(
        self,
        record_start: np.datetime64,
        record_end: np.datetime64,
    ):
        self.record_start = record_start
        self.record_end = record_end

        self._build_feature_tables()
        self._build_filter_table()

    # ------------------------------------------------------------------
    # FEATURE TABLE GENERATION
    # ------------------------------------------------------------------
    def _build_feature_tables(self):
        duration_sec = int(
            (self.record_end - self.record_start) / np.timedelta64(1, "s")
        )

        rng = np.random.default_rng(seed=42)

        n_rows_30 = max((duration_sec - 30) // self.SLIDING_WINDOW_STEP_SIZE, 1)
        n_rows_60 = max((duration_sec - 60) // self.SLIDING_WINDOW_STEP_SIZE, 1)

        self.features_30 = pd.DataFrame(
            rng.normal(size=(n_rows_30, len(self.FEATURE_NAMES_30))),
            columns=self.FEATURE_NAMES_30,
        )

        self.features_60 = pd.DataFrame(
            rng.normal(size=(n_rows_60, len(self.FEATURE_NAMES_60))),
            columns=self.FEATURE_NAMES_60,
        )


        # features_all: (n_windows, window_len, n_features)
        n_windows = max(
            (duration_sec - self.ALL_FEATURES_WINDOW_WIDTH)
            // self.SLIDING_WINDOW_STEP_SIZE,
            1,
        )
        window_len = self.ALL_FEATURES_WINDOW_WIDTH // self.SLIDING_WINDOW_STEP_SIZE

        self.features_all = rng.normal(
            size=(n_windows, window_len, self.get_n_features())
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # FILTER TABLE
    # ------------------------------------------------------------------
    def _build_filter_table(self):
        n_rows = self.features_all.shape[0]

        self.filter_table = pd.DataFrame({
            "without_nan": np.ones(n_rows, dtype=bool),
            "event_filter_1": np.ones(n_rows, dtype=bool),
            "without_seizure": np.ones(n_rows, dtype=bool),
            "label_seizures": np.zeros(n_rows, dtype=bool),
            "quality": np.ones(n_rows, dtype=int),
        })

    # ------------------------------------------------------------------
    # REQUIRED PUBLIC API
    # ------------------------------------------------------------------
    def get_prediction_data(self, event_filter_id: int, quality: int) -> np.ndarray:
        mask = self.filter_table["quality"] >= quality
        return self.features_all[mask]

    def get_prediction_labels(self, event_filter_id: int, quality: int) -> np.ndarray:
        mask = self.filter_table["quality"] >= quality
        return self.filter_table.loc[mask, "label_seizures"].to_numpy()

    def get_filtered_sample_evaluation_data(
        self, event_filter_id: int, quality: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.get_prediction_data(event_filter_id, quality),
            self.get_prediction_labels(event_filter_id, quality),
        )

    # ------------------------------------------------------------------
    # TIME ↔ ROW CONVERSION
    # ------------------------------------------------------------------
    def timestamp_to_row(self, timestamp: np.datetime64) -> int:
        offset_sec = utils.np_timedelta64_to_seconds(
            timestamp - self.record_start
        )
        return int(np.floor(offset_sec / self.SLIDING_WINDOW_STEP_SIZE))

    def row_to_timestamp(self, row: int) -> np.datetime64:
        return self.record_start + np.timedelta64(
            row * self.SLIDING_WINDOW_STEP_SIZE, "s"
        )

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def get_n_features(self) -> int:
        return (
            len(self.FEATURE_NAMES_30)
            + len(self.FEATURE_NAMES_60)
        )
