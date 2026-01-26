import os
import json
import logging
import RRcorrect  # Assuming `process_beat` function is in the `workers` modul

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
import sys
sys.path.append(r"/python-code/python-code/")

import monipy.utils.config as cfg
from ECGlabeling import ECGlabeling


# ---------------------------------------------------------------------------
# 1. Logging Setup
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("ecg_processing_logger")
LOGGER.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
LOGGER.addHandler(ch)

# File handler
fh = logging.FileHandler("ecg_processing.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
LOGGER.addHandler(fh)

# Prevent duplication in the root logger
LOGGER.propagate = False


# ---------------------------------------------------------------------------
# 2. Configuration and Constants
# ---------------------------------------------------------------------------



def get_default_config() -> Dict[str, float]:
    """
    Returns a dictionary of default configuration parameters
    for the R-peak correction process.
    """
    return {
        "P": 5,
        "Q": 3,
        "alpha": 0.02,
        "W": 60,
        "eta_e": 0.5,
        "eta_s": 3,
        "cpu": 3,       # example: number of processes
        "verbose": False,
    }

# Example constants for filtering, thresholds, etc.
THRESHOLD_LOWER_QUALITY = 10
THRESHOLD_UPPER_QUALITY = 75
MINIMUM_RR_INTERVAL_SEC = 0.3   # for negative-interval fixes
MAXIMUM_RR_INTERVAL_SEC = 2.0   # for extremely large intervals
SMALL_TIME_GAP_SEC = 5
PRE_WINDOW_SEC = get_default_config()["W"]
POST_WINDOW_SEC = 300
# ---------------------------------------------------------------------------
# 3. Manager Class for Handling Files and Data
# ---------------------------------------------------------------------------

class Manager:
    """
    Handles file/directory organization and seizure records metadata.
    """

    def __init__(self, main_folder: str) -> None:
        self.main_folder = main_folder
        self.location = os.path.basename(os.path.normpath(main_folder))
        self.seizures_info = self._load_seizures_info()
        self.seizures_ids = list(self.seizures_info.keys())

    def _load_seizures_info(self) -> Dict[str, Dict]:
        """
        Load global seizures_details.json for the folder.
        """
        file_path = os.path.join(self.main_folder, "seizures_details.json")
        with open(file_path, 'r') as file:
            seizures = json.load(file)
        return seizures

    def load_seizure_info(
        self,
        seizure_id: str,
        df: Optional[pd.DataFrame] = None,
        onset_column: str = "eeg_onset",
        offset_column: str = "eeg_offset",
        other: bool = False
    ) -> Tuple[np.datetime64, np.datetime64]:
        """
        Load either from the global JSON or from a DataFrame row (if other=True).
        """
        if other and df is not None:
            start = np.datetime64(df[df['seizure_id'] == seizure_id][onset_column].values[0])
            end = np.datetime64(df[df['seizure_id'] == seizure_id][offset_column].values[0])
            return start, end
        else:
            seizure_info_path = os.path.join(
                self.main_folder,
                self.seizures_info[seizure_id]['seizure_json'].replace("\\", "/")
            )
            with open(seizure_info_path, 'r') as file:
                seizure_info = json.load(file)
            return seizure_info

    def load_data(self, path: str):
        """
        Example: wrap ECGlabeling or custom data loader.
        """
        ecg_handler = ECGlabeling()
        ecg_handler.load_data(path)
        return ecg_handler

    def load_seizure(self, seizure_id: str, load_data: bool = False, data_type: str = "sensor_masked"):
        """
        Return seizure info and optionally load the ECG data file.
        """
        info = self.load_seizure_info(seizure_id)
        if load_data:
            data_path = info[data_type]
            ecg_data = self.load_data(data_path)
            return info, ecg_data
        else:
            return info
    def load_record(self, records, record_id, load_data=False):
        record_info = records[record_id]
        path = os.path.join(self.main_folder,record_info['unmasked'])
        if load_data:
            data = self.data_handler(path)
            return record_info, data
        else:
            return record_info

    def data_handler(self, path):
            ecg_handler = ECGlabeling()
            ecg_handler.load_data(path)
            return ecg_handler

    def get_rr_intervals(self, ecg_handler) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Compute RR intervals from raw r_peaks of ecg_handler.
        Returns: (RR intervals in ms, Timestamps of those intervals)
        """
        if not hasattr(ecg_handler, "r_peaks"):
            raise ValueError("ECG Handler has no 'r_peaks' attribute.")
        if len(ecg_handler.r_peaks) < 2:
            raise ValueError("Not enough R-peaks to compute intervals.")

        sampling_rate = ecg_handler.frequency
        start_time = pd.to_datetime(ecg_handler.ts[0])  # or ecg_handler.ts[0]
        r_peaks_seconds = ecg_handler.r_peaks / sampling_rate
        r_peaks_timedeltas = pd.to_timedelta(r_peaks_seconds, unit='s')
        r_peaks_datetimes = start_time + r_peaks_timedeltas
        rr_ms = np.diff(r_peaks_seconds) * 1000  # difference in sec -> ms

        return rr_ms, pd.DatetimeIndex(r_peaks_datetimes[1:])

    def get_all_record2(self, records, record_id):
        record_info, ecg_sen = self.load_record(records, record_id, load_data=True)
        r_peaks_seconds_sen, r_peaks_datetimes_sen = self.get_rr_intervals(ecg_sen)
        return record_info, r_peaks_seconds_sen, r_peaks_datetimes_sen, ecg_sen

    # More manager utilities as needed...


# ---------------------------------------------------------------------------
# 4. Helper Functions for Records & Seizures
# ---------------------------------------------------------------------------

def create_records(manage: Manager, exclude_patient_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    Creates unique record dicts from the Manager's seizures.
    Example structure:
      {
        "record_0": {
            "path_sensor": ...,
            "record_start_sen": ...,
            "record_end_sen": ...,
            "seizures": [...],
            "unmasked": ...,
            "patient_id": ...
        },
        ...
      }
    """
    records = {}
    exclude_patient_ids = set(exclude_patient_ids) if exclude_patient_ids else set()
    seizures_info = manage.seizures_info

    seen_records = set()
    idx = 0
    for sei_id in manage.seizures_ids:
        seizure = seizures_info[sei_id]
        patient_id = seizure["patient_id"]
        if patient_id in exclude_patient_ids:
            continue

        record_start_sen = seizure["record_start_sen"]
        if record_start_sen not in seen_records:
            seen_records.add(record_start_sen)
            records[f"record_{idx}"] = {
                "path_sensor": seizure["path_sensor"],
                "record_start_sen": record_start_sen,
                "record_end_sen": seizure["record_end_sen"],
                "seizures": [],
                "unmasked": None,
                "patient_id": patient_id
            }
            idx += 1

    # Assign seizure IDs to each record
    for s_id, s_info in seizures_info.items():
        for rec_key, rec_val in records.items():
            if rec_val["record_start_sen"] == s_info["record_start_sen"]:
                rec_val["seizures"].append(s_id)

    # Example: Mark "unmasked" from the last seizure in the record
    for rec_val in records.values():
        if rec_val["seizures"]:
            last_seizure = rec_val["seizures"][-1]
            rec_val["unmasked"] = os.path.join(last_seizure, "unmasked")

    return records


# ---------------------------------------------------------------------------
# 5. ECG Correction Logic
# ---------------------------------------------------------------------------

def generate_time_pairs(
    timestamps: List[np.datetime64],
    gap_sec: int = SMALL_TIME_GAP_SEC,
    pre_window: int = PRE_WINDOW_SEC,
    post_window: int = POST_WINDOW_SEC
) -> List[Tuple[np.datetime64, np.datetime64]]:
    """
    Identify continuous chunks of data by splitting where the gap
    between consecutive timestamps is > gap_sec. Then expand each chunk
    by `pre_window` seconds at the start and `post_window` seconds at the end.
    Finally, merge overlapping intervals to ensure no segment overlap.
    """
    time_pairs = []
    if len(timestamps) < 2:
        return time_pairs

    timestamps = np.array(timestamps)
    # Indices where gap is large
    gap_indices = np.where(np.diff(timestamps) > np.timedelta64(gap_sec, 's'))[0]
    gap_indices = np.append(gap_indices, len(timestamps) - 1)

    start_idx = 0
    for g_idx in gap_indices:
        # Expand window start/end
        window_start = timestamps[start_idx] - np.timedelta64(pre_window, "s")
        window_end = timestamps[g_idx] + np.timedelta64(post_window, "s")
        time_pairs.append((window_start, window_end))
        start_idx = g_idx + 1

    # Now merge overlapping or touching intervals in time_pairs
    merged_pairs = []
    # Sort by start time just in case (they should already be in order, but it's safe to do so).
    time_pairs.sort(key=lambda x: x[0])

    for interval in time_pairs:
        if not merged_pairs:
            merged_pairs.append(interval)
        else:
            prev_start, prev_end = merged_pairs[-1]
            curr_start, curr_end = interval

            # If current interval starts before (or exactly at) the end of previous, merge them
            if curr_start <= prev_end:
                merged_pairs[-1] = (prev_start, max(prev_end, curr_end))
            else:
                merged_pairs.append(interval)

    return merged_pairs


def prepare_arrays_for_correction(
    time_pairs: List[Tuple[np.datetime64, np.datetime64]],
    reference_start_time: pd.Timestamp,
    r_peaks: np.ndarray,
    sampling_rate: float
) -> List[np.ndarray]:
    """
    Convert r_peaks (in samples) to time-based arrays (in seconds),
    then split them according to time_pairs.
    """
    arrays_to_correct = []
    if len(r_peaks) < 2:
        return arrays_to_correct

    r_peaks_seconds = r_peaks / sampling_rate
    r_peaks_timestamps = reference_start_time + pd.to_timedelta(r_peaks_seconds, unit='s')

    for (start_t, end_t) in time_pairs:
        mask = (r_peaks_timestamps > start_t) & (r_peaks_timestamps < end_t)
        segment_indices = np.where(mask)[0]
        if len(segment_indices) > 0:
            # Convert to seconds
            segment = r_peaks_seconds[segment_indices]
            #segment = interpolate_rr_outliers(segment)
            arrays_to_correct.append(segment)

    return arrays_to_correct


def merge_corrected_segments(
    original_r_peaks_sec: np.ndarray,
    new_segments: List[Tuple[np.ndarray, Tuple[int, int]]],
    sampling_rate: float
) -> np.ndarray:
    """
    Replace each old segment in original_r_peaks_sec with the corrected version.
    new_segments is a list of tuples: (corrected_array, (start_index, end_index)).
    The start/end indices refer to slice in the *original array* to replace.
    """
    final_r_peaks = []
    last_idx = 0

    for corrected_array, (start_idx, end_idx) in new_segments:
        # Keep old data up to start_idx
        final_r_peaks.append(original_r_peaks_sec[last_idx:start_idx])
        # Insert corrected data
        final_r_peaks.append(corrected_array)
        last_idx = end_idx + 1

    # Append any remainder
    if last_idx < len(original_r_peaks_sec):
        final_r_peaks.append(original_r_peaks_sec[last_idx:])

    # Flatten
    if len(final_r_peaks) == 0:
        return original_r_peaks_sec
    return np.concatenate(final_r_peaks)


def merge_corrected_segments_with_debug(
    original_r_peaks_sec: np.ndarray,
    new_segments: list,
    min_rr: float = 0.3,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Merges corrected segments back into a final R-peak array, ensuring no
    negative intervals at segment boundaries. Logs detailed debug info about
    any forced shifts or overlaps.
    
    Parameters
    ----------
    original_r_peaks_sec : np.ndarray
        The original array of R-peaks in seconds (before segment corrections).
    new_segments : list of tuples
        Each tuple: (corrected_segment, (start_idx, end_idx))
          - corrected_segment is a *monotonically* increasing array of R-peaks (in seconds)
          - (start_idx, end_idx) are the indices in the *original* array that this segment replaces
    min_rr : float
        The forced minimum RR interval in seconds to avoid negative or zero intervals.
    logger : logging.Logger, optional
        A logger for debug messages. If None, uses a default logger or prints.

    Returns
    -------
    np.ndarray
        A single merged array of R-peaks in seconds, free of negative intervals
        at segment boundaries.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

    final_r_peaks = []
    last_end = 0  # track index in original_r_peaks_sec
    prev_last_peak = None  # track the last peak time from previous segment

    for seg_idx, (corrected_array, (start_idx, end_idx)) in enumerate(new_segments):
        # 1) Keep uncorrected data from last_end up to start_idx
        #    so we only replace [start_idx:end_idx + 1] with the corrected segment
        unchanged_slice = original_r_peaks_sec[last_end:start_idx]
        final_r_peaks.append(unchanged_slice)

        # 2) Now handle boundary with the previous segment
        if prev_last_peak is not None and len(corrected_array) > 0:
        #    # If the corrected segment's first peak is <= the last peak of previous segment,
        #    # shift it forward by min_rr
        #    if corrected_array[0] <= prev_last_peak:
        #        shifted_value = prev_last_peak + min_rr
        #        if logger:
        #            logger.warning(
        #                f"[merge_corrected] Segment {seg_idx}: shifting first peak from "
        #                f"{corrected_array[0]:.4f}s to {shifted_value:.4f}s to avoid overlap"
        #            )
        #        corrected_array[0] = shifted_value

            # If after shifting the first peak, we STILL detect negative intervals, log it again
            if corrected_array[0] <= prev_last_peak:
                if logger:
                    logger.error(
                        f"[merge_corrected] Segment {seg_idx}: FAILED to fix negative interval. "
                        f"First corrected peak {corrected_array[0]:.4f} <= prev_last_peak {prev_last_peak:.4f}. "
                        "Consider removing or adjusting segment further."
                    )

        # 3) Append the corrected segment
        final_r_peaks.append(corrected_array)

        # 4) Update last_end and prev_last_peak
        last_end = end_idx + 1
        if len(corrected_array) > 0:
            prev_last_peak = corrected_array[-1]
        else:
            # If the corrected array is empty, fallback to the last uncorrected peak
            # from the unchanged_slice if it exists
            if len(unchanged_slice) > 0:
                prev_last_peak = unchanged_slice[-1]

    # 5) Append any remainder from the original array (after the last segment’s end)
    if last_end < len(original_r_peaks_sec):
        final_r_peaks.append(original_r_peaks_sec[last_end:])

    # 6) Flatten the result
    if not final_r_peaks:
        # If no segments and no original data, just return original
        if logger:
            logger.warning("[merge_corrected] No final R-peaks found; returning original array.")
        return original_r_peaks_sec

    merged_r_peaks = np.concatenate(final_r_peaks)

    # 7) Final pass: check for negative intervals inside the merged array (just in case)
    #    And log exactly where it occurs
    negative_indices = np.where(np.diff(merged_r_peaks) < 0)[0]
    if len(negative_indices) > 0:
        if logger:
            logger.warning(
                f"[merge_corrected] Detected {len(negative_indices)} negative RR intervals after merging. "
                "Logging details of each issue..."
            )
        for idx in negative_indices:
            bad_peak = merged_r_peaks[idx]
            next_peak = merged_r_peaks[idx + 1]
            #if logger:
            #    logger.warning(
            #        f"  Negative interval at merged_r_peaks[{idx}..{idx+1}]: "
            #        f"{bad_peak:.4f}s -> {next_peak:.4f}s"
            #     )

    # Optionally, you can try another pass to fix them automatically
    # (be careful not to reintroduce them upstream).
    # For pure debugging, we just log them here.

    return merged_r_peaks


def interpolate_rr_outliers(
    r_peaks_sec: np.ndarray,
    rr_min: float = 0.3,
    rr_max: float = 2.0
) -> np.ndarray:
    """
    Identifies outlier RR intervals and interpolates to replace them.
    Parameters
    ----------
    r_peaks_sec : np.ndarray
        Array of R-peaks in seconds.
    rr_min : float
        Minimum acceptable RR interval (in sec).
    rr_max : float
        Maximum acceptable RR interval (in sec).
    Returns
    -------
    np.ndarray
        Cleaned R-peak array (in sec) with interpolated values replacing outlier intervals.
    """
    if len(r_peaks_sec) < 3:
        return r_peaks_sec.copy()

    r_peaks_clean = [r_peaks_sec[0]]
    for i in range(1, len(r_peaks_sec)):
        rr = r_peaks_sec[i] - r_peaks_clean[-1]
        if rr_min <= rr <= rr_max:
            r_peaks_clean.append(r_peaks_sec[i])
        else:
            # Replace with interpolated value
            interpolated_peak = r_peaks_clean[-1] + np.median(np.diff(r_peaks_sec))
            r_peaks_clean.append(interpolated_peak)

    return np.array(r_peaks_clean)


def apply_corrections_with_validation(
    config: Dict,
    time_pairs: List[Tuple[np.datetime64, np.datetime64]],
    ecg_handler,
    original_r_peaks_sec: np.ndarray,
    manage: Manager,
    rr_min: float = 0.3,
    rr_max: float = 2.5,
) -> np.ndarray:
    """
    Apply R-peak correction on segments and validate output.

    Parameters
    ----------
    config : dict
        Correction config.
    time_pairs : list of tuples
        Windows to process (start_time, end_time).
    ecg_handler : ECGlabeling
        Data object with r_peaks and ts.
    original_r_peaks_sec : np.ndarray
        R-peaks in seconds.
    manage : Manager
        Manager object for paths and metadata.
    rr_min : float
        Minimum allowable RR interval.
    rr_max : float
        Maximum allowable RR interval.

    Returns
    -------
    np.ndarray
        Final merged array of corrected R-peaks in seconds.
    """
    sampling_rate = ecg_handler.frequency
    start_time = pd.to_datetime(ecg_handler.ts[0])
    arrays_to_correct = prepare_arrays_for_correction(
        time_pairs, start_time, ecg_handler.r_peaks, sampling_rate
    )

    if not arrays_to_correct:
        LOGGER.warning("No valid segments found for correction.")
        return original_r_peaks_sec

    results = []
    with Pool(processes=config["cpu"]) as pool:
        tasks = [
            (
                seg, 
                config["P"], 
                config["W"], 
                config["alpha"], 
                config["eta_e"], 
                config["eta_s"], 
                config["Q"], 
                config["verbose"]
            )
            for seg in arrays_to_correct
        ]
        LOGGER.info("Starting parallel R-peak corrections...")
        for corrected_result in pool.imap(RRcorrect.process_actul, tasks):
            results.append(corrected_result)

    new_segments = []
    r_peaks_ts = start_time + pd.to_timedelta(original_r_peaks_sec, unit='s')
    segment_counter = 0

    for i, (start_t, end_t) in enumerate(time_pairs):
        mask = (r_peaks_ts > start_t) & (r_peaks_ts < end_t)
        seg_indices = np.where(mask)[0]

        if len(seg_indices) == 0:
            LOGGER.debug(f"Segment {segment_counter} skipped: no R-peaks in time window.")
            continue

        corrected_out = results[segment_counter]
        segment_counter += 1  # only increment here if segment was matched

        if corrected_out is None or len(corrected_out) < 2:
            LOGGER.warning(f"Segment {segment_counter-1} correction returned invalid — keeping original.")
            original_seg = original_r_peaks_sec[seg_indices[0]: seg_indices[-1] + 1]
            #original_seg = interpolate_rr_outliers(original_seg)
            new_segments.append((original_seg, (seg_indices[0], seg_indices[-1])))
            continue

        corrected_array = corrected_out[-1]

        # Validate monotonicity and physiological bounds
        rr = np.diff(corrected_array)
        #if not np.all(rr > 0):
        #    LOGGER.warning(f"Segment {i} skipped: non-monotonic corrected array — keeping original.")
        #    original_seg = original_r_peaks_sec[seg_indices[0]: seg_indices[-1] + 1]
        #    new_segments.append((original_seg, (seg_indices[0], seg_indices[-1])))
        #    continue
        #if np.any(rr < rr_min) or np.any(rr > rr_max):
        #    LOGGER.warning(f"Segment {i} skipped: RR interval out of physiological range — keeping original.")
        #    original_seg = original_r_peaks_sec[seg_indices[0]: seg_indices[-1] + 1]
        #    new_segments.append((original_seg, (seg_indices[0], seg_indices[-1])))
        #    continue

        # Accepted
        new_segments.append((corrected_array, (seg_indices[0], seg_indices[-1])))
        LOGGER.info(f"Segment {segment_counter-1} accepted: len={len(corrected_array)}, RR mean={np.mean(rr):.3f}")

    # Merge segments
    merged_r_peaks_sec = merge_corrected_segments_with_debug(
        original_r_peaks_sec,
        new_segments,
        min_rr=rr_min,
        logger=LOGGER
    )

    # Debug plot
    debug_plot_dir = os.path.join(manage.main_folder, "debug_plots")
    os.makedirs(debug_plot_dir, exist_ok=True)
    save_path = os.path.join(debug_plot_dir, f"corrected_segments_plot{start_time}.png")
    plot_corrected_segments(
        original_r_peaks_sec=original_r_peaks_sec,
        corrected_r_peaks_sec=merged_r_peaks_sec,
        new_segments=new_segments,
        time_pairs=time_pairs,
        sampling_rate=ecg_handler.frequency,
        save_path=save_path
    )

    return merged_r_peaks_sec




def plot_corrected_segments(
    original_r_peaks_sec: np.ndarray,
    corrected_r_peaks_sec: np.ndarray,
    new_segments: list,
    time_pairs: list,
    sampling_rate: float,
    save_path: str
):
    """
    Plots only the corrected segments (not the entire time series),
    showing RR-intervals against an x-axis in seconds relative to start_time.

    Parameters
    ----------
    original_r_peaks_sec : np.ndarray
        The array of original R-peaks in seconds (relative to start_time).
    corrected_r_peaks_sec : np.ndarray
        The array of corrected R-peaks in seconds (relative to start_time).
    new_segments : list of tuples
        Each tuple: (corrected_segment_array, (start_idx, end_idx)).
        - corrected_segment_array is the new R-peaks for that segment
        - (start_idx, end_idx) are the indices in the original array that got replaced
    time_pairs : list of tuples
        Each tuple: (start_time, end_time) in np.datetime64 or pd.Timestamp,
        corresponding to the corrected segment's time window.
    sampling_rate : float
        The sampling rate of the ECG in Hz (not strictly needed here but included for context).
    save_path : str
        File path to save the resulting plot (PNG, etc.).
    """
    num_segments = len(new_segments)
    if num_segments == 0:
        print("No corrected segments to plot.")
        return

    # Create a figure large enough to hold multiple subplots
    # For example, 2 columns, so number of rows is (num_segments + 1)//2
    n_cols = 2
    n_rows = (num_segments + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), dpi=100)

    # If there's only 1 row or 1 axis, ensure consistent array shape
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])  # shape (1,1)
    elif n_rows == 1:
        axes = np.array([axes])    # shape (1,n_cols)

    axes_flat = axes.flatten()  # so we can index easily in a loop

    for i, ((new_seg, (start_idx, end_idx)), (win_start, win_end)) in enumerate(zip(new_segments, time_pairs)):
        if i >= len(axes_flat):
            break  # safety check if we have more segments than subplots
        ax = axes_flat[i]

        # Original R-peaks in this segment
        orig_slice = original_r_peaks_sec[start_idx : end_idx + 1]
        # Corrected R-peaks for this segment
        corrected_slice = new_seg

        # Compute RR intervals for both
        orig_rr = np.diff(orig_slice)
        corr_rr = np.diff(corrected_slice)

        # For the x-axis, place each RR interval at the midpoint of the two R-peaks.
        # This means if we have peaks p0 < p1 < p2 ..., each RR interval covers (p1-p0), (p2-p1), ...
        # The midpoint times for these intervals would be (p0+p1)/2, (p1+p2)/2, etc.
        if len(orig_slice) > 1:
            x_orig = (orig_slice[:-1] + orig_slice[1:]) / 2.0
        else:
            x_orig = []

        if len(corrected_slice) > 1:
            x_corr = (corrected_slice[:-1] + corrected_slice[1:]) / 2.0
        else:
            x_corr = []

        # Plot them
        ax.plot(x_orig, orig_rr, label="Original RR")
        ax.plot(x_corr, corr_rr, linestyle="--", label="Corrected RR")

        # Format the title and labels
        start_str = str(pd.to_datetime(win_start))[-8:]  # e.g., "HH:MM:SS"
        end_str = str(pd.to_datetime(win_end))[-8:]
        ax.set_title(f"Segment {i} ({start_str} - {end_str})")

        ax.set_xlabel("Time (s) relative to start_time")
        ax.set_ylabel("RR Interval (s)")

        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Plot saved to: {save_path}")



def plot_comparison(
    original_r_peaks_sec: np.ndarray,
    corrected_r_peaks_sec: np.ndarray,
    sampling_rate: float,
    save_path: str,
    time_pairs: list = None,
    recording_start_time: pd.Timestamp = None
) -> None:
    """
    Plot a single-figure comparison of original vs. corrected RR intervals.
    Highlights time intervals (from time_pairs) on the x-axis in seconds.

    Parameters
    ----------
    original_r_peaks_sec : np.ndarray
        Original R-peak times in seconds (relative to 'recording_start_time').
    corrected_r_peaks_sec : np.ndarray
        Corrected R-peak times in seconds (relative to 'recording_start_time').
    sampling_rate : float
        ECG sampling rate, not strictly needed here but included for context.
    save_path : str
        File path to save the resulting plot (PNG, etc.).
    time_pairs : list of tuples, optional
        Each tuple: (start_time, end_time), where start_time and end_time are
        np.datetime64 or pd.Timestamp representing “good-quality” intervals.
        Defaults to None (no highlighting).
    recording_start_time : pd.Timestamp, optional
        The reference start time of the recording. Needed to convert time_pairs
        into seconds. If None, no highlighting is done (or you must handle
        conversions differently).
    """
    try:
        # Safety check
        if len(original_r_peaks_sec) < 2 or len(corrected_r_peaks_sec) < 2:
            print("Insufficient R-peaks to plot comparison.")
            return

        # 1) Compute RR intervals
        original_rr = np.diff(original_r_peaks_sec)
        corrected_rr = np.diff(corrected_r_peaks_sec)

        # 2) Midpoint for each RR interval => x-axis in seconds
        x_orig = (original_r_peaks_sec[:-1] + original_r_peaks_sec[1:]) / 2.0
        x_corr = (corrected_r_peaks_sec[:-1] + corrected_r_peaks_sec[1:]) / 2.0

        # 3) Make sure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 4) Plot on a single figure
        plt.figure(figsize=(10, 5))

        # Plot RR intervals
        plt.plot(x_orig, original_rr, label="Original RR")
        plt.plot(x_corr , corrected_rr, linestyle="--", label="Corrected RR")

        plt.xlabel("Time (s) from start")
        plt.ylabel("RR Interval (s)")
        plt.title("Comparison of Original vs. Corrected RR Intervals")
        plt.legend()

        # 5) Highlight the good-quality intervals
        if time_pairs and recording_start_time is not None:
            for (seg_start_dt, seg_end_dt) in time_pairs:
                # Convert each datetime to "seconds from the recording_start_time"
                start_sec = (pd.to_datetime(seg_start_dt) - recording_start_time).total_seconds()
                end_sec   = (pd.to_datetime(seg_end_dt)   - recording_start_time).total_seconds()

                # Shade this time range with a semi-transparent red rectangle
                plt.axvspan(start_sec, end_sec, color="red", alpha=0.5)

        # 6) Save and close
        plt.ylim(0.0, 5.5)
        plt.xlim(start_sec-1000, end_sec)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"Saved comparison plot to: {save_path}")

    except Exception as e:
        print(f"Error in plot_comparison: {e}")
        plt.close()

def save_corrected_ecg(
    manage: Manager,
    seizure_info: Dict,
    ecg_handler,
    corrected_r_peaks_sec: np.ndarray
) -> None:
    """
    Saves the corrected R-peaks back into an ECG file or structure.
    Adjust the path if needed.
    """
    unmasked_path = os.path.join(manage.main_folder, seizure_info["unmasked"])
    try:
        # Re-load to be sure we're overwriting the correct file
        updated_handler = ECGlabeling()
        updated_handler.load_data(unmasked_path)

        updated_handler.r_peaks = corrected_r_peaks_sec * updated_handler.frequency
        updated_handler.save_data()

        LOGGER.info(f"Successfully saved corrected ECG data to {unmasked_path}")
    except Exception as e:
        LOGGER.error(f"Failed to save corrected ECG data to {unmasked_path}: {e}")

