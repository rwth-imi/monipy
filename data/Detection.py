"""This class defines the Detection objects to store detections made by the models during evaluation"""
import numpy as np


class Detection:
    def __init__(
        self,
        record_id: int,
        timestamp_start: np.datetime64,
        timestamp_end: np.datetime64,
    ) -> None:

        self.record_id = record_id
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self._status = "unknown"

    def __str__(self):
        dur = (
            (self.timestamp_end - self.timestamp_start)
            .astype("timedelta64[s]")
            .astype(int)
        )
        return f"[{self._status}] Record_id: {self.record_id}, {self.timestamp_start} - {self.timestamp_end} -> {dur} secs"

    def set_status(self, status: str) -> None:
        if status not in ["tp", "fp", "other_seizure","bad_quality"]:
            raise ValueError(f'{status} is not a valid status, only "tp" or "fp"')
        self._status = status

    def get_status(self) -> str:
        return self._status
