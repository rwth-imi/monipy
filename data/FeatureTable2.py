"""This class defines a tabular structure to store the features computed on a record"""
import numpy as np
import pandas as pd
import os.path
## test
import json
import traceback
from typing import List, Union, Tuple, Dict, Any
from sqlalchemy.engine import Connectable
from sqlalchemy import text
import monipy.utils.features as features
import monipy.utils.utils as utils
from monipy.data.Record2 import Record2
from monipy.data.EventFilter import EventFilter
from monipy.data.RRISection import RRISection
import os


def compute_and_save_features(args):
    compute_func, filename, filepath = args
    if not os.path.isfile(filename):
        print(f"Computing {filepath}")
        features = compute_func()
        print(f"Done with {filepath}")
        features.to_pickle(filepath)
        del features


class FeatureTable:
    """Holds the features for one record

    .. todo:: Add detailed description & usage instructions

    The last few rows will be all nan most of the times. This is due to the r-peak detection algorithm we used. It often
    skips the last few minutes...

    Attributes:
        record_id (int): Record_id for this FeatureTable

    """

    FILENAME_FEATURES_30 = "features_gen2_30.pkl"
    FILENAME_FEATURES_60 = "features_gen2_60.pkl"
    FILENAME_FEATURES_120 = "features_gen2_120.pkl"
    FILENAME_FEATURES_ALL = "features_gen2_all.npy"
    FILENAME_FITER_TABLE = "filter_gen2_table.pkl"

    def __init__(
        self,
        records: None,
        record_id: None,
        manage: None,
        onset: None,
        info: None,
        verbose : bool = False,
        force_creation: bool = False,
        partial_load: List[str] = [
            "features_30",
            "features_60",
            "features_120",
            "features_all",
            "filter_table",
        ],
        compute_ft = True,
        con=None,
    ) -> None:
        """Creates a FeatureTable for given record_id

        Tries to load the features from files if existing, if not the features are computed.

        Args:
            record_id (int): Record_id
        """
        seizure = manage.load_record(records, record_id, load_data=False)
        self.record_path = os.path.join(manage.main_folder,seizure['unmasked'])
        self.record_info = seizure
        #self._log("Data was loaded")
        self.SLIDING_WINDOW_STEP_SIZE = 5  # in secs
        self.ALL_FEATURES_WINDOW_WIDTH = 5 * 60  # in secs
        self.manage = manage
        self.onset = onset
        self.info = info
        self.verbose = verbose
        self.record_id = record_id
        self._set_filepaths()
        self.copmute_ft = compute_ft

        # load or compute

        if self._all_feature_files_exist() and not force_creation:
            #self._log(f"[{self.record_id}]: feature files found, loading them")
            self._load_features(partial_load)
        else:
            try :
                seizure, r_peaks, ts_rpeaks = manage.get_all_record(records, record_id)
                #del _
                self.record_path = os.path.join(manage.main_folder,seizure['unmasked'])
                self.record_info = seizure
                seizure["timestamp_start"] = np.datetime64(ts_rpeaks[0])
                seizure["timestamp_end"] = np.datetime64(ts_rpeaks[-1])
                self._log("Data was loaded")
                #self.ecg_sen = ecg_sen

                self.record = Record2(file_name=None, header=seizure, device_specific_header=None,
                                      rpeak_indices=r_peaks, rpeak_classes=np.array([0]), rpeak_timestamps=ts_rpeaks,
                                      intervals=RRISection(r_peaks, ts_rpeaks[0:-1].to_numpy(), ts_rpeaks[1:].to_numpy()))
                self._log("Record2 was created")

                del r_peaks
                del ts_rpeaks
                
                self._log("Computing 30")
                self.features_30 = self._compute_features_30()
                self._log("done with 30")
                self.features_30.to_pickle(self._filepath_30)
                #del self.features_30

                self._log("Computing 60")
                self.features_60 = self._compute_features_60()
                self._log("done with 60")
                self.features_60.to_pickle(self._filepath_60)
                del self.features_60
                
                self._log("Computing 120")
                self.features_120 = self._compute_features_120()
                self._log("done with 120")
                self.features_120.to_pickle(self._filepath_120)
                del self.features_120

                self._log("Computing all")
                partial_load: List[str] = [
                    "features_30",
                    "features_60",
                    "features_120"
                ]
                self._load_features(partial_load)
                self.features_all = self._compute_features_all()
                self._log("done with all")
                np.save(self._filepath_all, self.features_all)

                self.filter_table = self._compute_filter_table(con=con)
                self._log("done with filter table")
                self.filter_table.to_pickle(self._filepath_filter)
                #self._save_features()
                self._log(f"done with {self.record_id}")

                # except RuntimeError as err:
            except ValueError as err:
                # Todo: Log it? If yes, where?
                print(f"Could not compute FeatureTable for record {self.record_id}")
                print(f"Corresponding error: {err}")
                print("-------\n")
                traceback.print_exc()




    def _save_features(self) -> None:
        """Saves the features to files

        Returns:

        """
        self.features_30.to_pickle(self._filepath_30)
        self.features_60.to_pickle(self._filepath_60)
        self.features_120.to_pickle(self._filepath_120)
        np.save(self._filepath_all, self.features_all)
        self.filter_table.to_pickle(self._filepath_filter)

    def _load_features(self, partial_load: List[str]) -> None:
        """Loads the features from files

        Returns:

        """

        # Maurice: Sorry for lazy implementation, it's almost weekend ¯\_(ツ)_/¯

        if "features_30" in partial_load:
            self.features_30 = pd.read_pickle(self._filepath_30)
        if "features_60" in partial_load:
            self.features_60 = pd.read_pickle(self._filepath_60)
        if "features_120" in partial_load:
            self.features_120 = pd.read_pickle(self._filepath_120)
        if "features_all" in partial_load:
            self.features_all = np.load(self._filepath_all)
        if "filter_table" in partial_load:
            self.filter_table = pd.read_pickle(self._filepath_filter)

    def check_for_event_filter(self, event_filter_id: int) -> None:
        """
        Checks if a certain event filter is present in the filter table

        Args:
            event_filter_id (int): ID of the Event FIlter as in database.

        Raises:
            RuntimeError: Event FIlter is not present in Filter Table.

        Returns:
            None.

        """
        event_filter_column_name = f"event_filter_{event_filter_id}"
        if event_filter_column_name not in self.filter_table.columns:
            raise RuntimeError(
                (
                    f"feature table for record {self.record_id} (file"
                    f"{self.record_path}) has no column {event_filter_column_name}"
                )
            )

    def get_feature_data(
        self,
        timestamp_start: np.datetime64,
        duration: np.timedelta64,
        return_nans=False,
    ) -> pd.DataFrame:
        """
        Fetch feature data for a given timestamp and duration

        Args:
            timestamp_start (np.datetime64): Start timestamp to fetch the seizures.
            duration (np.timedelta64): Over which duration the features should be fetched.
            return_nans (TYPE, optional): Return feature rows including nans. Defaults to False.

        Raises:
            IndexError: Timestamp and/or duration is not feasible.
            ValueError: Duration is invalid.
            RuntimeError: Feature data has nans and they should not be returned.

        Returns:
            feature_data (DataFrame): DataFrame containing the features for the given time frame.

        """

        # checks
        if timestamp_start < self.record.header["timestamp_start"]:
            raise IndexError("Requested timestamp_start is before record start.")

        if timestamp_start + duration > self.record.header["timestamp_end"]:
            raise IndexError(
                "Requested timestamp_start + duration is after record end."
            )

        if duration < np.timedelta64(120, "s"):
            raise ValueError("duration has to be at least 120 secs")

        if not utils.np_timedelta64_to_seconds(duration) % 5 == 0:
            raise ValueError("duration needs to be a multiple of 5 seconds")

        # Start-row is the same for all Features
        start_row = self.timestamp_to_row(timestamp_start)

        # some readability
        n_rows_30, n_rows_60, n_rows_120 = self._get_n_rows(duration)
        n_features_30 = self.features_30.shape[1]
        n_features_60 = self.features_60.shape[1]
        n_features_120 = self.features_120.shape[1]

        # create matrix to store all (interpolated) data
        interpolated_data = np.nan * np.zeros(
            (n_rows_30, n_features_30 + n_features_60 + n_features_120)
        )

        # features_30 is fine, just copy the data
        interpolated_data[:, 0:n_features_30] = self.features_30.iloc[
            start_row : start_row + n_rows_30
        ].values

        # features_60 need to be interpolated
        feature_data_60_raw = self.features_60.iloc[
            start_row : start_row + n_rows_60
        ].values
        new_x_60 = np.linspace(0, n_rows_60 - 1, num=n_rows_30)
        for i in range(n_features_60):
            data_old = feature_data_60_raw[:, i]
            data_new = np.interp(new_x_60, range(n_rows_60), data_old)
            interpolated_data[:, n_features_30 + i] = data_new

        # features_120 need to be interpoleted in the same way
        feature_data_120_raw = self.features_120.iloc[
            start_row : start_row + n_rows_120
        ].values
        new_x_120 = np.linspace(0, n_rows_120 - 1, num=n_rows_30)
        for i in range(n_features_120):
            data_old = feature_data_120_raw[:, i]
            data_new = np.interp(new_x_120, range(n_rows_120), data_old)
            interpolated_data[:, n_features_30 + n_features_60 + i] = data_new

        feature_data = pd.DataFrame(
            interpolated_data, columns=self._get_all_feature_names()
        )

        if return_nans or not feature_data.isnull().values.any():
            return feature_data
        else:
            print("-------------")
            print("Error on fetching feature data from")
            print(f"Record {self.record_id}")
            print(f"Start {timestamp_start}")
            print(f"Duration {duration}")
            print(f"Return_nans {return_nans}")
            print("fetched feature data:")
            #display(feature_data)
            print("filter table:")
            #display(self.filter_table[start_row : start_row + n_rows_30])

            raise RuntimeError("feature_data has nans.")

    def check_if_all_sampling_windows_are_good(
        self,
        sampling_time: np.datetime64,
        sampling_windows: np.ndarray,
        event_filter_id: int,
        quality: int,
        verbose: bool = False,
    ) -> bool:
        """Checks if all windows relative to sampling_time are passing the event filter and don't have
        NaNs in the feature data

        Args:
            sampling_time (np.datetime64): Timestamp where to sample (for seizures: Seizure onset)
            sampling_windows (np.ndarray): Relative to sampling_time, in 5sec steps
            event_filter_id (int): Event Filter ID

        Returns:

        """

        # row of sampling_time
        base_row = self.timestamp_to_row(sampling_time)

        # sampling_windows is already in 5sec steps = row indices
        sampling_rows = base_row + sampling_windows

        try:

            # check event filter
            event_filter_is_good = self.filter_table.iloc[sampling_rows][
                f"event_filter_{event_filter_id}"
            ].any()
            utils.bool_verbose_print(
                event_filter_is_good,
                verbose,
                "Not all windows are passing the event filter",
            )

            # check without nan
            data_is_without_nan = self.filter_table.iloc[sampling_rows]["without_nan"].all()
            utils.bool_verbose_print(
                data_is_without_nan, verbose, "In some windows are NaNs in feature data"
            )

            # Check data quality
            data_is_good = (self.filter_table.iloc[sampling_rows]['quality'] >= quality).all()
            utils.bool_verbose_print(
                data_is_good, verbose, "In some windows, the quality is below the threshold"
            )

            
            return event_filter_is_good and data_is_without_nan and data_is_good

        except IndexError:
            return False

    def check_if_all_sampling_windows_are_good_unpacked(
        self,
        sampling_time: np.datetime64,
        sampling_windows: np.ndarray,
        event_filter_id: int,
        quality: int,
        verbose: bool = False,
    ) -> bool:
        """Checks if all windows relative to sampling_time are passing the event filter and don't have
        NaNs in the feature data

        Args:
            sampling_time (np.datetime64): Timestamp where to sample (for seizures: Seizure onset)
            sampling_windows (np.ndarray): Relative to sampling_time, in 5sec steps
            event_filter_id (int): Event Filter ID

        Returns:

        """

        # row of sampling_time
        base_row = self.timestamp_to_row(sampling_time)

        # sampling_windows is already in 5sec steps = row indices
        sampling_rows = base_row + sampling_windows

        try:

            # check event filter
            event_filter_is_good = self.filter_table.iloc[sampling_rows][
                f"event_filter_{event_filter_id}"
            ].any()
            utils.bool_verbose_print(
                event_filter_is_good,
                verbose,
                "Not all windows are passing the event filter",
            )

            # check without nan
            data_is_without_nan = self.filter_table.iloc[sampling_rows]["without_nan"].all()
            utils.bool_verbose_print(
                data_is_without_nan, verbose, "In some windows are NaNs in feature data"
            )

            # Check data quality
            data_is_good = (self.filter_table.iloc[sampling_rows]['quality'] >= quality).all()
            utils.bool_verbose_print(
                data_is_good, verbose, "In some windows, the quality is below the threshold"
            )

            return event_filter_is_good , data_is_without_nan, data_is_good

        except IndexError:
            return False, False, False

    def get_sampling_window_features(
        self,
        sampling_time: np.datetime64,
        sampling_windows: np.ndarray,
        event_filter_id: int,
        quality: int,
    ) -> List[Dict[str, Any]]:
        """Returns feature data for all sampling_windows relative to sampling_time

        Args:
            sampling_time (np.datetime64): Timestamp where to sample (for seizures: Seizure onset)
            sampling_windows (np.ndarray): Relative to sampling_time, in 5sec steps
            event_filter_id (int): Event Filter ID

        Returns:
            feature_data_list (List[pd.DataFrame])

        """

        if not self.check_if_all_sampling_windows_are_good(
            sampling_time, sampling_windows, event_filter_id, quality
        ):
            raise RuntimeError("Not all requested sampling windows are good!")

        feature_data_list = []
        for sampling_window_offset in sampling_windows:
            sample_timestamp = sampling_time + np.timedelta64(
                sampling_window_offset * 5, "s"
            )
            feature_data = self.get_feature_data(
                timestamp_start=sample_timestamp, duration=np.timedelta64(300, "s")
            )

            # sanity checks
            if len(feature_data.columns) != 38:
                raise RuntimeError(
                    f"Invalid column number({len(feature_data.columns)}) for record {self.record_id}"
                )

            feature_data_list.append(
                {
                    "timestamp_start": sample_timestamp,
                    "feature_data": feature_data,
                    "window_index": sampling_window_offset,
                }
            )

        return feature_data_list

    def get_filtered_sample_evaluation_data(
        self, event_filter_id: int,  quality: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return data and labels for a given event filter

        Args:
            event_filter_id (int): Event Filter ID as in database.

        Returns:
            features (np.ndarray): Feature data.
            labels (np.ndarray): Labels.

        """
        features = self.get_prediction_data(event_filter_id, quality)
        labels = self.get_prediction_labels(event_filter_id, quality)
        return features, labels

    def get_prediction_data(self, event_filter_id: int, quality: int) -> np.ndarray:
        """
        Fetch input features for a given event filter, after applying quality filtering.

        Args:
            event_filter_id (int): Event Filter ID as in database.
            quality (int): Minimum quality threshold.

        Returns:
            features (np.ndarray): Feature data.
        """
        event_filter_column_name = f"event_filter_{event_filter_id}"
        
        # Step 1: Filter for quality first
        quality_mask = self.filter_table['quality'] >= quality
        filtered_table = self.filter_table[quality_mask]

        # Step 2: Then filter for event
        event_idc = np.array(
            filtered_table.index[filtered_table[event_filter_column_name] == True]
        )
        
        return self.features_all[event_idc, :]

    def get_prediction_labels(self, event_filter_id: int, quality: int) -> np.ndarray:
        """
        Fetch labels for a given event filter

        Args:
            event_filter_id (int): Event Filter ID as in database.

        Returns:
            features (np.ndarray): Labels.

        """
        event_filter_column_name = f"event_filter_{event_filter_id}"
       # Step 1: Filter for quality first
        quality_mask = self.filter_table['quality'] >= quality
        filtered_table = self.filter_table[quality_mask]

        # Step 2: Then filter for event
        event_idc = np.array(
            filtered_table.index[filtered_table[event_filter_column_name] == True]
        )
        
        #temp = self.filter_table["without_seizure"].iloc[event_idc].to_numpy()

        return self.filter_table["label_seizures"].iloc[event_idc].to_numpy()
    def _compute_features_30(self) -> pd.DataFrame:
        """
        Computes the features defined on a 30 second rolling window

        Raises:
            RuntimeError: The record does not contain enough data.

        Returns:
            df (DataFrame): Features30.

        """
        features_30 = [
            ["avg", features.avg],
            ["sd", features.sd],
            ["rmssd", features.rmssd],
            ["rmssd_dt", features.rmssd_dt],
            ["skew", features.skew],
            ["kurt", features.kurt],
            ["pnnx", features.pnnx],
            ["nnx", features.nnx],
            ["triangular_index", features.triangular_index],
            ["quantile_25", features.quantile_25],
            ["quantile_50", features.quantile_50],
            ["quantile_75", features.quantile_75],
            ["variance", features.variance],
        ]

        # create empty matrix with the correct size to store all feature windows
        try:  # todo: why try?
            feature_matrix = np.nan * np.empty(
                (self._get_feature_matrix_rows(30), len(features_30)), np.float32
            )
        except ValueError:
            raise RuntimeError

        # loop in 5 sek schritten
        i = 0
        window_start = self.record.header["timestamp_start"]
        window_width = np.timedelta64(30, "s")
        total_iterations = (self.record.header["timestamp_end"] - window_start) // np.timedelta64(
            self.SLIDING_WINDOW_STEP_SIZE, "s")
        # Replace the while loop with a tqdm loop for the progress bar
        print("Computing features 30")
        for _ in range(total_iterations):
            rri = self.record.get_rri_slice(
                window_start, window_start + window_width
            ).get_raw_rri_values()

            if np.sum(rri) < (30 - 5) * 1000:
                rri = np.nan

            # some features dont work with nans, so skip if we have some
            if not np.isnan(np.sum(rri)):
                for k, feature in enumerate(features_30):
                    feature_matrix[i, k] = feature[1](rri)

            i += 1
            window_start += np.timedelta64(self.SLIDING_WINDOW_STEP_SIZE, "s")

        df_columns = [f[0] for f in features_30]
        df = pd.DataFrame(feature_matrix, columns=df_columns)

        return df

    def _compute_features_60(self) -> pd.DataFrame:
        """
        Computes the features defined on a 60 second rolling window

        Returns:
            df (DataFrame): Features60.

        """
        df_columns = [
            "csi",
            "csim",
            "cvi",
            "sd1",
            "sd2",
            "csi_slope",
            "csim_slope",
            "csi_filtered",
            "csim_filtered",
            "csi_filtered_slope",
            "csim_filtered_slope",
            "hr_diff",
            "hr_diff_slope",
            "hr_diff_filtered",
            "hr_diff_filtered_slope",
        ]

        # create empty matrix with the correct size to store all feature windows
        feature_matrix = np.nan * np.empty(
            (self._get_feature_matrix_rows(60), len(df_columns)), np.float32
        )

        # loop in 5 sek schritten
        i = 0
        window_start = self.record.header["timestamp_start"]
        window_width = np.timedelta64(60, "s")
        total_iterations = (self.record.header["timestamp_end"] - window_start) // np.timedelta64(
            self.SLIDING_WINDOW_STEP_SIZE, "s")
        # Replace the while loop with a tqdm loop for the progress bar
        print("Computing features 60")
        for _ in range(total_iterations):
            rri = self.record.get_rri_slice(
                window_start, window_start + window_width
            ).get_raw_rri_values()

            if np.sum(rri) < (30 - 5) * 1000:
                rri = np.array([np.nan])

            # some features dont work with nans, so skip if we have some

            if not np.isnan(np.sum(rri)) and len(rri) > 10:
                #print(np.sum(rri))
                # 0-4: Poincare Features
                poincare_features = features.poincare(rri)
                feature_matrix[i, 0] = poincare_features["csi"]
                feature_matrix[i, 1] = poincare_features["csim"]
                feature_matrix[i, 2] = poincare_features["cvi"]
                feature_matrix[i, 3] = poincare_features["sd1"]
                feature_matrix[i, 4] = poincare_features["sd2"]

                # 5-10: Poincare Filtered Slope
                poincare_features_fs = features.poincare_filtered_slope(
                    rri, poincare_features
                )
                feature_matrix[i, 5] = poincare_features_fs["csi_slope"]
                feature_matrix[i, 6] = poincare_features_fs["csim_slope"]
                feature_matrix[i, 7] = poincare_features_fs["csi_filtered"]
                feature_matrix[i, 8] = poincare_features_fs["csim_filtered"]
                feature_matrix[i, 9] = poincare_features_fs["csi_filtered_slope"]
                feature_matrix[i, 10] = poincare_features_fs["csim_filtered_slope"]

                # 11-14: HR Diff
                hr_diff_features = features.hr_diff_all(rri)
                feature_matrix[i, 11] = hr_diff_features["hr_diff"]
                feature_matrix[i, 12] = hr_diff_features["hr_diff_slope"]
                feature_matrix[i, 13] = hr_diff_features["hr_diff_filtered"]
                feature_matrix[i, 14] = hr_diff_features["hr_diff_filtered_slope"]

            i = i + 1
            window_start = window_start + np.timedelta64(
                self.SLIDING_WINDOW_STEP_SIZE, "s"
            )


        df = pd.DataFrame(feature_matrix, columns=df_columns)

        return df

    def _compute_features_120(self) -> pd.DataFrame:
        """
        Computes the features defined on a 120 second rolling window

        Raises:
            RuntimeError: The record does not contain enough data.

        Returns:
            df (DataFrame): Features120.

        """
        df_columns = [
            "ulf",
            "vlf",
            "lf",
            "hf",
            "lf_hf_ratio",
            "total_power",
            "mf_hurst_max",
            "mf_coef_left",
            "mf_coef_center",
            "mf_coef_right",
        ]

        # create empty matrix with the correct size to store all feature windows
        feature_matrix = np.nan * np.empty(
            (self._get_feature_matrix_rows(120), len(df_columns)), np.float32
        )

        # loop in 5 sek schritten
        i = 0
        window_start = self.record.header["timestamp_start"]
        window_width = np.timedelta64(120, "s")
        total_iterations = (self.record.header["timestamp_end"] - window_start) // np.timedelta64(
            self.SLIDING_WINDOW_STEP_SIZE, "s")
        # Replace the while loop with a tqdm loop for the progress bar
        print("Computing features 120")
        for _ in range(total_iterations):
            rri = self.record.get_rri_slice(
                window_start, window_start + window_width
            ).get_raw_rri_values()

            if np.sum(rri) < (30 - 5) * 1000:
                rri = np.nan

            # some features dont work with nans, so skip if we have some
            if not np.isnan(np.sum(rri)):
                freq_features = features.freq_based(rri)
                mf_features = features.get_multifractal_spectrum_trends(rri)

                feature_matrix[i, 0] = freq_features["ulf"]
                feature_matrix[i, 1] = freq_features["vlf"]
                feature_matrix[i, 2] = freq_features["lf"]
                feature_matrix[i, 3] = freq_features["hf"]
                feature_matrix[i, 4] = freq_features["lf_hf_ratio"]
                feature_matrix[i, 5] = freq_features["total_power"]

                feature_matrix[i, 6] = mf_features[0]
                feature_matrix[i, 7] = mf_features[1]
                feature_matrix[i, 8] = mf_features[2]
                feature_matrix[i, 9] = mf_features[3]

            i = i + 1
            window_start = window_start + np.timedelta64(
                self.SLIDING_WINDOW_STEP_SIZE, "s"
            )

        df = pd.DataFrame(feature_matrix, columns=df_columns)

        return df

    def _compute_features_all(self) -> np.ndarray:
        """
        Compute features_all, the np arrayy containing features for all roling window sizes chained

        Returns:
            ndarray: features_all.

        """
        window_width = np.timedelta64(self.ALL_FEATURES_WINDOW_WIDTH, "s")
        step_size = np.timedelta64(self.SLIDING_WINDOW_STEP_SIZE, "s")
        record_start = self.record.header["timestamp_start"]

        first_feature_window = self.get_feature_data(
            record_start, window_width, return_nans=True
        )
        n_rows = self._get_feature_matrix_rows(self.ALL_FEATURES_WINDOW_WIDTH)

        feature_matrix = np.empty(
            (n_rows, first_feature_window.shape[0], first_feature_window.shape[1]),
            np.float32,
        )

        for i in range(n_rows):
            feature_df = self.get_feature_data(
                record_start + i * step_size, window_width, return_nans=True
            )
            feature_matrix[i, :, :] = feature_df.values

        return feature_matrix.astype(np.float32)

    def get_patient_seizures(self,current):
        current_seizure_patient = self.manage.seizures_info[current]["patient_id"]
        alls = list()
        for sei in self.manage.seizures_ids:
            if self.manage.seizures_info[sei]["patient_id"] == current_seizure_patient:
                alls.append(sei)
        return np.array(alls)

    def get_true_start(self,seizure_id, con):
        in_sei = int(float(seizure_id))
        #org_record_id = database.get_seizure_record(in_sei, device_type="movisens", con=con)
        org_record_id = database.get_seizure_record(in_sei, con=con)
        seizure_start, _ = database.get_seizure_timestamps(in_sei, record_id=org_record_id, con=con)
        return seizure_start

    def get_seizures(self,con):
        ids = self.record_info["seizures"]
        near = list()
        for id in ids:
            main = np.datetime64(self.get_true_start(id, con=con))
            near.append(main)
        return np.array(near)

    def _compute_filter_table(self, con: Connectable) -> pd.DataFrame:
        """
        Compute the filter table

        Args:
            con (Connectable): Connection to the database.

        Returns:
            filter_table (DataFrame): Filter Table.

        """
        query = text("SELECT id, config FROM `event_filters`")
        event_filters = pd.read_sql(query, con=con)
        seizure_starts = self.get_seizures(con=con)


        columns = (
            ["without_nan"]
            + [f"event_filter_{i}" for i in event_filters.id]
            + ["without_seizure"]
        )
        n_rows_30, n_rows_60, n_rows_120 = self._get_n_rows(np.timedelta64(300, "s"))
        n_windows = len(self.features_30) - n_rows_30

        raw_data = np.ones((n_windows, len(columns)), dtype=np.bool)

        for index in np.arange(n_windows):
            if self.copmute_ft:
                if not self._check_without_nans(index, n_rows_30, n_rows_60, n_rows_120):
                    raw_data[index, :] = False
                    continue

            data_window = self.features_30.iloc[index : index + n_rows_30]

            # Apply all event filters in the database
            raw_data[index, 1:-1] = self._apply_event_filters(
                data_window, event_filters.config
            )

            # Search for seizures in proximity, only check if
            if np.any(raw_data[index, 1:-1]):
                raw_data[index, -1] = not self._check_seizure_proximity(
                    seizure_starts, index
                )
            else:
                raw_data[index, -1] = False

        filter_table = self._compute_label_cps(
            pd.DataFrame(raw_data, columns=columns), con=con
        )

        return filter_table

    def _check_without_nans(
        self, index: int, n_rows_30: int, n_rows_60: int, n_rows_120: int
    ) -> bool:
        """
        Helper function for _compute_filter_table

        Args:
            index (int): Inde to be checked.
            n_rows_30 (int): Number of rows to check for features30.
            n_rows_60 (int): Number of rows to check for features60.
            n_rows_120 (int) Number of rows to check for features120.

        Returns:
            bool: Does the data contain nan values at the specified index.

        """
        # Couldve done this more pythonic but chose readability instead
        nans_30 = self.features_30.iloc[index : index + n_rows_30].isnull().values.any()
        nans_60 = self.features_60.iloc[index : index + n_rows_60].isnull().values.any()
        #nans_120 = (
        #    self.features_120.iloc[index : index + n_rows_120].isnull().values.any()
        #)

        if np.any([nans_30, nans_60]):
            return False

        return True

    @staticmethod
    def _apply_event_filters(
        data_window: pd.DataFrame, filter_configs: pd.Series
    ) -> List[bool]:
        """
        Helper function for _compute_filter_table

        Args:
            data_window (pd.DataFrame): Slice containing feature data.
            filter_configs (pd.Series): Configurations for Event Filters.

        Returns:
            list[bool]: List of entries indicating if the data window passes the given Event Filter.

        """
        return [
            EventFilter(data_window, json.loads(filter_config)).is_event
            for filter_config in filter_configs
        ]

    def _check_seizure_proximity(self, seizure_starts: np.ndarray, index: int) -> bool:
        """
        Helper function for _compute_filter_table

        Args:
            seizure_starts (np.ndarray): Array containing seizure start timestamps.
            index (int): Index to be checked for seizure proximity.

        Returns:
            bool: Flag indicating if a seizure is close to index.

        """
        if len(seizure_starts) == 0:
            return False

        return np.any(
            np.abs(
                seizure_starts
                - (np.datetime64(self.record.first_interval_start) + np.timedelta64(5, "s") * index)
            )
            <= np.timedelta64(15, "m")
        )
    def _compute_label_seizures(
        self, filter_table: pd.DataFrame, con: Connectable
    ) -> pd.DataFrame:
        """
        Compute the seizures label in the filter table

        Args:
            filter_table (pd.DataFrame): Filter Table.
            con (Connectable): Connection to the database.

        Returns:
            filter_table (DataFrame): Updated Filter Table.

        """
        #TODO make sure this shit is correct
        offset_left = np.timedelta64(-90, "s")
        offset_right = np.timedelta64(300, "s")

        filter_table["label_seizures"] = False
        seizure_starts = self.get_seizures(con=con)
        # update labels for all seizures
        for seizure_start in seizure_starts:


            idx_from = self.timestamp_to_row(seizure_start + offset_left)
            idx_to = self.timestamp_to_row(seizure_start + offset_right)
            #print(seizure_start, idx_from, idx_to)
            filter_table["label_seizures"].iloc[idx_from:idx_to] = True

        return filter_table
    
    def _get_feature_matrix_rows(self, feature_width: int) -> int:
        """Tells us how many rows we need to store all windows from the sliding window

        Args:
            feature_width (int): width of the features in seconds

        Returns:
            int: number of rows needed for the feature matrix
        """
        result = np.floor(
            (
                self.record.get_duration_in_seconds()
                - feature_width
                + self.SLIDING_WINDOW_STEP_SIZE
            )
            / self.SLIDING_WINDOW_STEP_SIZE
        ).astype("int")

        if result <= 0:
            raise ValueError("The record does not contain enough data")

        return result

    def _get_all_feature_names(self) -> List[str]:
        """
        Return all features currently calculated

        Returns:
            List[str]: Feature list.

        """
        return (
            self.features_30.columns.to_list()
            + self.features_60.columns.to_list()
            + self.features_120.columns.to_list()
        )

    def _set_filepaths(self):
        """
        Helper ethod to set file paths

        Returns:
            None.

        """
        self._filepath_30 = os.path.join(self.record_path[:-9], self.FILENAME_FEATURES_30)
        self._filepath_60 = os.path.join(self.record_path[:-9], self.FILENAME_FEATURES_60)
        self._filepath_120 = os.path.join(self.record_path[:-9], self.FILENAME_FEATURES_120)
        self._filepath_all = os.path.join(self.record_path[:-9], self.FILENAME_FEATURES_ALL)
        self._filepath_filter = os.path.join(
            self.record_path[:-9], self.FILENAME_FITER_TABLE
        )

    def _all_feature_files_exist(self) -> bool:
        """Checks if all feature table files exist in the record folder

        Returns:
            bool: true if all files exist, false if at least one is missing
        """
        files_to_check = [
            self.FILENAME_FEATURES_30,
            self.FILENAME_FEATURES_60,
            self.FILENAME_FEATURES_120,
            self.FILENAME_FEATURES_ALL,
            self.FILENAME_FITER_TABLE,
        ]

        for file in files_to_check:
            if not os.path.isfile(os.path.join(self.record_path[:-9], file)):
                # todo: check file size??
                return False

        return True


    def timestamp_to_row(self, timestamp_start: np.datetime64) -> int:
        """
        Convert timestamp to row in features all / filter table

        Args:
            timestamp_start (np.datetime64): Timestamp to find corresponding row for.

        Returns:
            int: Row in features_all / Filter Table.

        """
        offset_sec = utils.np_timedelta64_to_seconds(
            timestamp_start - self.record.header["timestamp_start"]
        )
        return np.floor(offset_sec / self.SLIDING_WINDOW_STEP_SIZE).astype(int)

    def row_to_timestamp(self, row: int) -> np.datetime64:
        return self.record.header["timestamp_start"] + np.timedelta64(
            row * self.SLIDING_WINDOW_STEP_SIZE, "s"
        )

    def _get_n_rows(self, duration: np.timedelta64) -> List[int]:
        """
        Helper method to find rows needed to model duration with rolling window of 30, 60, 120 seconds

        Args:
            duration (np.timedelta64): Duration to be modeled.

        Returns:
            List[int]: n_rows for 30,60,120 second window size.

        """
        return [
            int(
                (utils.np_timedelta64_to_seconds(duration) - window_width)
                / self.SLIDING_WINDOW_STEP_SIZE
            )
            + 1
            for window_width in [30, 60, 120]
        ]

    def _log(self, message: str):
            print(message)

    @staticmethod
    @database.create_connection
    def load_features_30(record_id: int, con=None) -> pd.DataFrame:
        return FeatureTable._save_load_sub_data(
            record_id, FeatureTable.FILENAME_FEATURES_30, con=con
        )

    @staticmethod
    @database.create_connection
    def load_features_60(record_id: int, con=None) -> pd.DataFrame:
        return FeatureTable._save_load_sub_data(
            record_id, FeatureTable.FILENAME_FEATURES_60, con=con
        )

    @staticmethod
    @database.create_connection
    def load_features_120(record_id: int, con=None) -> pd.DataFrame:
        return FeatureTable._save_load_sub_data(
            record_id, FeatureTable.FILENAME_FEATURES_120, con=con
        )

    @staticmethod
    @database.create_connection
    def load_features_all(record_id: int, con=None) -> np.ndarray:
        return FeatureTable._save_load_sub_data(
            record_id, FeatureTable.FILENAME_FEATURES_ALL, con=con
        )

    @staticmethod
    @database.create_connection
    def load_filter_table(record_id: int, con=None) -> pd.DataFrame:
        return FeatureTable._save_load_sub_data(
            record_id, FeatureTable.FILENAME_FITER_TABLE, con=con
        )
