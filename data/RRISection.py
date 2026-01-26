from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from typing import Tuple, Union


class RRISection:
    def __init__(
        self,
        intervals_len: np.ndarray,
        intervals_left_ts: np.ndarray,
        intervals_right_ts: np.ndarray,
    ):
        self.intervals_len = intervals_len  # in ms
        self.intervals_left_ts = intervals_left_ts.astype(
            "datetime64[ms]"
        )  # in ms resolution
        self.intervals_right_ts = intervals_right_ts.astype(
            "datetime64[ms]"
        )  # in ms resolution

        # there are cases where we need empty sections
        if self.intervals_len.size > 0:
            self.start = self.intervals_left_ts[0]
            self.end = self.intervals_right_ts[-1]
            self.size = self.intervals_len.shape[0]
            self.duration = self.end - self.start
        else:
            self.start = None
            self.end = None
            self.size = 0
            self.duration = np.timedelta64(0, "s")

    def get_slice(
        self,
        slice_from: Union[np.datetime64, int],
        slice_to: Union[np.datetime64, int],
        check_consistency: bool = False,
        check_boundaries: bool = True,
    ) -> RRISection:
        """Return subsection

        Args:
            slice_from (): Can be np.datetime or int index
            slice_to (): Can be np.datetime or int index
            check_consistency ():

        Returns:

        """

        int_types = [int, np.int64]
        datetime_types = [np.datetime64, pd.Timestamp]

        if type(slice_from) in datetime_types and type(slice_to) in datetime_types:
            from_idx, to_idx = self.get_slice_indices(
                slice_from, slice_to, check_consistency, check_boundaries
            )

        elif type(slice_from) in int_types and type(slice_to) in int_types:
            from_idx = slice_from
            to_idx = slice_to

        else:
            raise TypeError(
                "slice_from and slice_to needs to be BOTH either int or np.datetime64! "
                f"Given: {type(slice_from)}, {type(slice_to)}"
            )

        # print(f"[RRISec.get_slice]:from: {from_idx}, to: {to_idx}")

        if from_idx > to_idx:
            return RRISection(np.array([]), np.array([]), np.array([]))

        return RRISection(
            self.intervals_len[from_idx:to_idx],
            self.intervals_left_ts[from_idx:to_idx],
            self.intervals_right_ts[from_idx:to_idx],
        )

    def get_slice_indices(
        self,
        slice_from: np.datetime64,
        slice_to: np.datetime64,
        check_consistency: bool = False,
        check_boundaries: bool = True,
    ) -> Tuple[np.int64, np.int64]:
        """returns the indices needed for slicing

        Args:
            slice_from ():
            slice_to ():

        Returns:

        """
        # checks
        if check_consistency:
            if not self._is_valid():
                raise RuntimeError("RRISection is invalid!")

        if check_boundaries:
            if slice_from < self.intervals_left_ts[0]:
                raise IndexError(
                    f"slice_from ({slice_from}) is before first interval ({self.intervals_left_ts[0]})"
                )

            if slice_to > self.intervals_right_ts[-1]:
                raise IndexError(
                    f"slice_to ({slice_to}) is after last interval ({self.intervals_right_ts[-1]})"
                )

        if slice_from > slice_to:
            raise IndexError("slice_from has to be before slice_to")

        # get slice indices
        from_idx = np.min(np.argwhere(self.intervals_left_ts >= slice_from))
        to_idx = np.max(np.argwhere(self.intervals_right_ts <= slice_to))

        return from_idx.astype(np.int64), to_idx.astype(np.int64)

    def get_raw_rri_values(self):
        return self.intervals_len.copy()

    def split_one_interval_in_two(self, i: int, new_value_left: int = None) -> None:
        """Splits the RR-Interval at index i in two. If new_value_left is none, the old interval will be splittet into
        two equal sizes intervals. If not, the first will be new_value left and the second old_value - new_value_left

        Args:
            i ():
            new_value_left ():

        Returns:

        """

        # TODO: boundary checks

        if new_value_left is None:
            new_value_left = int(round(self.intervals_len[i] / 2))

        new_second = self.intervals_len[i] - new_value_left

        self.intervals_len[i] = new_value_left
        self.intervals_len = np.insert(self.intervals_len, i + 1, new_second)

        old_left = self.intervals_left_ts[i]
        old_right = self.intervals_right_ts[i]
        new_ts = old_left + np.timedelta64(new_value_left, "ms")

        self.intervals_right_ts[i] = new_ts

        self.intervals_left_ts = np.insert(self.intervals_left_ts, i + 1, new_ts)
        self.intervals_right_ts = np.insert(self.intervals_right_ts, i + 1, old_right)

        self.size = self.size + 1

    def split_one_interval_into_many(
        self, idx: int, fill_values: np.ndarray, check_consistency: bool = True
    ) -> None:
        old_left_ts = self.intervals_left_ts[idx]

        self.intervals_len[idx] = fill_values[0]
        self.intervals_right_ts[idx] = old_left_ts + np.timedelta64(
            fill_values[0], "ms"
        )

        insert_left_ts = np.zeros(fill_values.size - 1, dtype="datetime64[ms]")
        insert_right_ts = np.zeros(fill_values.size - 1, dtype="datetime64[ms]")

        delta = fill_values[0]

        for k in range(fill_values.size - 1):
            insert_left_ts[k] = old_left_ts + np.timedelta64(int(delta), "ms")
            delta = delta + fill_values[k + 1]
            insert_right_ts[k] = old_left_ts + np.timedelta64(int(delta), "ms")

        self.intervals_len = np.insert(self.intervals_len, idx + 1, fill_values[1:])
        self.intervals_left_ts = np.insert(
            self.intervals_left_ts, idx + 1, insert_left_ts
        )
        self.intervals_right_ts = np.insert(
            self.intervals_right_ts, idx + 1, insert_right_ts
        )

        self.size = self.size + fill_values.size - 1

        if check_consistency and not self._is_valid():
            raise RuntimeError("non valid RRISegment!")

    def join_two_intervals(self, i: int) -> None:
        """Joins the intervals at i and i+1 to one single interval

        Args:
            i ():

        Returns:

        """

        # TODO: boundary checks

        # i+1 = i+1 + i+2, remove i+2
        self.intervals_len[i] = self.intervals_len[i] + self.intervals_len[i + 1]
        self.intervals_right_ts[i] = self.intervals_right_ts[i + 1]

        self.intervals_len = np.delete(self.intervals_len, i + 1)
        self.intervals_left_ts = np.delete(self.intervals_left_ts, i + 1)
        self.intervals_right_ts = np.delete(self.intervals_right_ts, i + 1)

        self.size = self.size - 1

    def get_raw_rri_values_with_nans(self, min_rri: int, max_rri: int):
        # todo implement
        raise NotImplementedError("maybe we don't need this function")

    def plot(self, title: str = "", show: bool = True):
        if self.size > 10000:
            print(
                "plotly cant handle so much data. Please plot only a subsection of this RRISection "
                "(by using get_slice())"
            )
            return

        fig = px.line(x=self.intervals_left_ts, y=self.intervals_len)

        # fixed axis for better comparison
        fig.update_yaxes(range=[0, 2000])

        fig.update_layout(title_text=title)

        if show:
            fig.show()
        else:
            return fig

    @staticmethod
    def from_rpeak_timestamps(rpeak_timestamps: np.ndarray) -> RRISection:
        """Creates RRISection based on given array of timestamps of r-peaks

        Args:
            rpeak_timestamps ():

        Returns:

        """
        if not rpeak_timestamps.dtype == "<M8[ms]":
            raise RuntimeError(
                f"dtype of rpeak_timestmaps needs to be '<M8[ms]' (is {rpeak_timestamps.dtype})"
            )

        intervals_len = np.diff(rpeak_timestamps).astype(np.float64)
        intervals_left_ts = rpeak_timestamps[0:-1]
        intervals_right_ts = rpeak_timestamps[1:]

        return RRISection(intervals_len, intervals_left_ts, intervals_right_ts)

    def _is_valid(self) -> bool:
        """Tells us if:
        - self has at least one interval
        - dimensions of all fields are correct
        - all values are consistent

        Returns:

        """

        # not empty?
        if (
            (self.intervals_len.shape[0] == 0)
            or (self.intervals_right_ts.shape[0] == 0)
            or (self.intervals_left_ts.shape[0] == 0)
        ):
            print("something is empty")
            return False

        # same size
        if (not self.intervals_len.shape == self.intervals_left_ts.shape) or (
            not self.intervals_left_ts.shape == self.intervals_right_ts.shape
        ):
            print("not same size")
            return False

        # check all values are consistent
        for i in range(self.intervals_len.shape[0]):

            # check if len matches difference of timestamps
            if not np.isnan(self.intervals_len[i]):
                ts_delta = np.timedelta64(
                    self.intervals_right_ts[i] - self.intervals_left_ts[i]
                ).astype("int")
                if not self.intervals_len[i] == ts_delta:
                    print(
                        f"inconsistend data at index {i}: len = {self.intervals_len[i]}ms, ts_delta = {ts_delta}ms"
                    )
                    return False

            # check if right[i] == left[i+1]
            if i < self.intervals_len.shape[0] - 1:
                if not self.intervals_right_ts[i] == self.intervals_left_ts[i + 1]:
                    print(
                        f"timestamps at index {i}/{i+1} do not match! "
                        f"({self.intervals_right_ts[i]} vs. {self.intervals_left_ts[i-1]})"
                    )
                    return False
        # check attributes
        if not self.intervals_len.size == self.size:
            print("size is incorrect")
            return False

        return True
