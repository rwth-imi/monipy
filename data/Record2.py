from __future__ import annotations

import numpy as np
import pandas as pd
import os.path
import pickle

from monipy.data.RRISection import RRISection



class Record2:
    def __init__(
        self,
        file_name: str,
        header: dict,
        device_specific_header: dict,
        rpeak_indices: np.ndarray,
        rpeak_classes: np.ndarray,
        rpeak_timestamps: np.ndarray,
        intervals: RRISection,
    ):

        self.header = header
        self.device_specific_header = device_specific_header
        self.rpeak_indices = rpeak_indices
        self.rpeak_timestamps = rpeak_timestamps  # in ms resolution!
        self.rpeak_classes = (
            rpeak_classes  # 0= rpdetection, 1= filter/interp, 2= manual
        )

        self.intervals = intervals
        self.source_file = file_name

        self.first_interval_start = self.intervals.intervals_left_ts[0]
        self.last_interval_end = self.intervals.intervals_right_ts[-1]



    def get_rri_slice(
        self, slice_from: np.datetime64, slice_to: np.datetime64
    ) -> RRISection:
        try:
            rri_slice = self.intervals.get_slice(slice_from, slice_to)
        except Exception as e:
            print(f"Record.get_rri_slice Exception: {e}. Returning empty slice")
            rri_slice = RRISection(
                np.array([np.nan]), np.array([slice_from]), np.array([slice_to])
            )

        return rri_slice

    def save(self, filename: str) -> None:
        """Saves the record to given filename

        Args:
            filename ():

        Returns:

        """

        # prevent rp.mat files from overwriting
        _, ext = os.path.splitext(filename)
        if ext == ".mat":
            raise RuntimeError("Never save a Record2 object to a .mat file!")

        # self.source_file is only locally valid. Hence it should not be saved in the file.
        temp = self.source_file
        self.source_file = None

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        self.source_file = temp


    def get_duration_in_seconds(self) -> int:
        """Returns the duration of the record (time between header.timestamp_start and header.timestamp_end) in seconds

        Returns:
            int: the duration in seconds
        """

        delta = np.timedelta64(
            self.header["timestamp_end"] - self.header["timestamp_start"], "s"
        ).astype("int")
        return delta
