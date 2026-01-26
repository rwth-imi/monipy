import numpy as np
from typing import Dict, Any


class SeizureEvaluation:
    def __init__(
        self,
        record_id: int,
        seizure_id: int,
        timestamp_start: np.datetime64,
        timestamp_end: np.datetime64,
        data: Dict[str, Any],
    ) -> None:

        self.record_id = record_id
        self.seizure_id = seizure_id
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.data = data

        self.is_correct_type = False
        self.has_valid_data = False
        self.is_event = False
        self.is_detectable = False
        self.is_detected = False

    def __str__(self):
        return (
            f"Seizure_id: {self.seizure_id}, Record_id: {self.record_id}, "
            f"is_correct_type: {self.is_correct_type}  "
            f"is_detectable(hvd/ie): {self.has_valid_data}/{self.is_event} -> {self.is_detectable}  "
            f"is_detected: {self.is_detected}"
        )
