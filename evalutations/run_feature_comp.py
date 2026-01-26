# imports 
import json
import os
import numpy as np
import pandas as pd

from ECGlabeling import ECGlabeling
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# for pretty printing
w1 = 87
w2 = 43
w3 = 21
w4 = 10
linesep = "-" * 89
import sys

# Create a logger

# Create a named logger
logger = logging.getLogger('ecg_processing_logger')
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create file handler and set level to debug
fh = logging.FileHandler('ecg_processing.log')
fh.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to handlers
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add handlers to loggerQUALITY_LOWER
logger.addHandler(ch)
logger.addHandler(fh)

# Prevent log messages from propagating to the root logger
logger.propagate = False

# Example log message
logger.debug("This is a debug message for ECG processing.")



def plot_rpeak_comparison(ecg_sen, record_id, save_dir="quality_plots", threshold=1500):
    """
    Plots 10 random 200-second ECG + quality slices with correct R-peak overlays.

    Parameters:
        ecg_sen: ECGlabeling instance with .raw, .ts, .frequency, .r_peaks
        record_id: str
        save_dir: str
        threshold: int (optional, used in quality())
    """
    os.makedirs(save_dir, exist_ok=True)
    record_id_dir = os.path.join(save_dir, f"{record_id}")
    os.makedirs(record_id_dir, exist_ok=True)

    fs = ecg_sen.frequency
    total_samples = len(ecg_sen.raw)
    seconds_window = 50
    samples_window = int(seconds_window * fs)

    timestamps = np.array(ecg_sen.ts, dtype='datetime64[ns]')
    qual, auto1 = quality(ecg_sen, fs)
    window_seconds = 5
    quality_start_time = timestamps[0] + np.timedelta64(5, 's')
    new_timestamps = pd.date_range(
        start=quality_start_time,
        periods=len(qual),
        freq='5S'
    )

    r_peak_indices = ecg_sen.r_peaks.astype(int)
    r_peak_times = timestamps[r_peak_indices]
    r_peak_amps = ecg_sen.raw[r_peak_indices]


    auto1_indices = auto1.astype(int)
    auto1_times = timestamps[auto1_indices]
    auto1_amps = ecg_sen.raw[auto1_indices]


    # Pick random start indices
    max_start = total_samples - samples_window
    random_start_indices = np.random.choice(max_start, size=10, replace=False)

    for i, start_idx in enumerate(random_start_indices):
        end_idx = start_idx + samples_window

        time_slice = timestamps[start_idx:end_idx]
        signal_slice = ecg_sen.raw[start_idx:end_idx]

        # Filter R-peaks inside the slice
        mask = (r_peak_times >= time_slice[0]) & (r_peak_times <= time_slice[-1])
        r_ham_times = r_peak_times[mask]
        r_ham_amps = r_peak_amps[mask]

        mask1 = (auto1_times >= time_slice[0]) & (auto1_times <= time_slice[-1])
        r_ham_times1 = auto1_times[mask1]
        r_ham_amps1 = auto1_amps[mask1]

         # Plot RR intervals
        rr_intervals = np.diff(r_ham_times)
        if np.issubdtype(rr_intervals.dtype, np.timedelta64):
            rr_intervals = rr_intervals / np.timedelta64(1, 's')

        # Plot RR intervals
        rr_intervals1 = np.diff(r_ham_times1)
        if np.issubdtype(rr_intervals1.dtype, np.timedelta64):
            rr_intervals1 = rr_intervals1 / np.timedelta64(1, 's')

        fig, ax = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

        # Plot ECG signal
        ax[0].plot(time_slice, signal_slice/np.max(np.abs(signal_slice)), alpha=0.6, label="ECG") 
        ax[0].scatter(r_ham_times, r_ham_amps / np.max(np.abs(signal_slice)), color='g', label='R-peaks')

        ax[0].legend()
        ax[0].set_ylabel("Normalized ECG")

    
        ax[1].plot(r_ham_times[:-1], rr_intervals,"-o", color='g', label='Hamilton Intervals')
        ax[1].legend()
        ax[1].set_ylabel("RR")
        ax[1].set_xlabel("Time (s)")

        ax[2].plot(r_ham_times1[:-1], rr_intervals1,"-o", color='g', label='SSF Intervals')
        ax[2].legend()
        ax[2].set_ylabel("RR")
        ax[2].set_xlabel("Time (s)")


        qual_mask = (new_timestamps >= time_slice[0]) & (new_timestamps <= time_slice[-1])
        sliced_quality = qual[qual_mask]
        sliced_quality_times = new_timestamps[qual_mask]
        # Plot Quality
        ax[3].plot(sliced_quality_times, sliced_quality / 100, label="Quality")
        ax[3].legend()
        ax[3].set_xlim(time_slice[0], time_slice[-1])
        ax[3].set_ylim(-.1, 1.2)
        ax[3].set_ylabel("Quality")
        ax[3].set_xlabel("Time (s)")


        # Set main title
        fig.suptitle(f"ECG + R-peaks + Quality – Record {record_id}, Slice {i}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(record_id_dir, f"{record_id}_slice_{i}.png")
        plt.savefig(output_path)
        plt.close()
    return qual


### fucntions 
class Manager:

    def __init__(self, main_folder) -> None:
        self.main_folder = main_folder
        self.location = self.main_folder.split("\\")[-1]
        self.seizures_info = self.load_seizures_info()
        self.seizures_ids = list(self.seizures_info.keys())

    def load_seizures_info(self):
        file_path = os.path.join(self.main_folder, "seizures_details.json")
        with open(file_path, 'r') as file:
            seizures = json.load(file)
        return seizures

    def load_seizure_info(self, seizure_id, df=None, onset=None, offset=None, other=False):
        if other:
            start = np.datetime64(df[df['seizure_id'] == seizure_id][onset].values[0])
            end = np.datetime64(df[df['seizure_id'] == seizure_id][offset].values[0])
            return start, end
        else:
            seizure_info_path = os.path.join(self.main_folder,self.seizures_info[seizure_id]['seizure_json'].replace("\\","/"))
            with open(seizure_info_path, 'r') as file:
                seizure_info = json.load(file)
            return seizure_info

    def data_handler(self, path):
        ecg_handler = ECGlabeling()
        ecg_handler.load_data(path)
        return ecg_handler

    def load_seizure(self, seizure_id, load_data=False, types="sensor_masked"):
        info = self.load_seizure_info(seizure_id)
        if load_data:
            data = self.data_handler(info[types])
            return info, data
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

    def get_rr(self, ecg):
        mask_timestamps_series = pd.to_datetime(ecg.ts)
        sampling_rate = ecg.frequency
        print(sampling_rate)
        start_time = mask_timestamps_series[0]
        r_peaks_seconds = ecg.r_peaks / sampling_rate
        r_peaks_timedeltas = pd.to_timedelta(r_peaks_seconds, unit='s')
        r_peaks_datetimes = start_time + r_peaks_timedeltas
        return np.diff(r_peaks_seconds) * 1000, r_peaks_datetimes[1:]

    def get_all(self, selected_seizure):
        seizure, ecg_sen = self.load_seizure(selected_seizure, load_data=True)
        r_peaks_seconds_sen, r_peaks_datetimes_sen = self.get_rr(ecg_sen)
        return seizure, r_peaks_seconds_sen, r_peaks_datetimes_sen

    def get_all_record(self, records, record_id):
        record_info, ecg_sen = self.load_record(records, record_id, load_data=True)
        r_peaks_seconds_sen, r_peaks_datetimes_sen = self.get_rr(ecg_sen)
        return record_info, r_peaks_seconds_sen, r_peaks_datetimes_sen

    def get_all_record2(self, records, record_id):
        record_info, ecg_sen = self.load_record(records, record_id, load_data=True)
        r_peaks_seconds_sen, r_peaks_datetimes_sen = self.get_rr(ecg_sen)
        return record_info, r_peaks_seconds_sen, r_peaks_datetimes_sen, ecg_sen


def get_patients_ids(df_seizures,manage):
    all_ps = list()
    for sei in df_seizures:
        p =  manage.seizures_info[sei]["patient_id"]
        all_ps.append(p)
    return np.array(list(set(all_ps)))
def create_records(manage, exclude_patient_ids=None):
    records = {}

    # Load seizure info once
    seizures_info = manage.seizures_info  
    exclude_patient_ids = set(exclude_patient_ids) if exclude_patient_ids else set()

    # First pass: Create unique records
    seen_records = set()
    
    for ind, sei in enumerate(manage.seizures_ids):
        seizure = seizures_info[sei]
        record_start_sen = seizure["record_start_sen"]
        record_end_sen = seizure["record_end_sen"]
        patient_id = seizure["patient_id"]
        orig_path = seizure["path_sensor"]
        
        if patient_id in exclude_patient_ids:
            continue
        # Avoid duplicate record creation
        if record_start_sen not in seen_records:
            seen_records.add(record_start_sen)
            records[f"record_{ind}"] = {
                'path_sensor': orig_path,
                "record_start_sen": record_start_sen,
                "record_end_sen": record_end_sen,
                "seizures": [],
                "bad_seizure": [],
                "other_seizure": [],
                "patient_id": patient_id,
                "unmasked": None,  # Placeholder to avoid recomputation
            }

    # Second pass: Load seizure info once and assign seizures to records
    all_seizures_info = manage.load_seizures_info()
    
    for s, seizure_info in all_seizures_info.items():
        record_start_sen = seizure_info["record_start_sen"]
        
        for rec in records.values():
            if rec["record_start_sen"] == record_start_sen:
                rec["seizures"].append(s)

    # Ensure "unmasked" points to the last seizure in the list
    for rec in records.values():
        if rec["seizures"]:  # Ensure list is not empty
            rec["unmasked"] = os.path.join(rec["seizures"][-1], "unmasked")  # Last seizure as unmasked

    # Assign "unmasked" path to all seizures in the record
    for rec in records.values():
        for sei in rec["seizures"]:
            manage.load_seizure_info(sei)["unmasked"] = rec["unmasked"]

    return records


def match_ori(ori, ec, thresh):
    """
    Match ori spikes to ec spikes within a time difference threshold (in samples).
    """
    matched_ori = np.zeros(len(ori), dtype=bool)
    matched_ec = np.zeros(len(ec), dtype=bool)

    # Compute absolute differences between all peak pairs
    diff_matrix = np.abs(ori[:, np.newaxis] - ec[np.newaxis, :])
    close_matches = diff_matrix <= thresh

    for i in range(len(ori)):
        valid_matches = np.where(close_matches[i] & ~matched_ec)[0]
        if valid_matches.size > 0:
            closest_idx = valid_matches[np.argmin(diff_matrix[i, valid_matches])]
            matched_ori[i] = True
            matched_ec[closest_idx] = True

    matched_count = np.sum(matched_ori)
    unmatched_ori = np.sum(~matched_ori)
    unmatched_ec = np.sum(~matched_ec)

    return matched_count, unmatched_ori, unmatched_ec


def bSQI(detector_1, detector_2, fs=1024, mode="n_double", search_window=150):
    """
    Compare two detectors' peak outputs using beat-by-beat matching.
    search_window: in milliseconds
    """
    if detector_1 is None or detector_2 is None:
        raise TypeError("Input Error, check detectors outputs")

    if len(detector_1) == 0 or len(detector_2) == 0:
        return 0.0

    # Convert ms to sample threshold
    thresh_samples = int((search_window / 1000) * fs)
    tp, _, _ = match_ori(detector_1, detector_2, thresh_samples)

    both = tp
    if mode == "simple":
        return (both / len(detector_1)) * 100
    elif mode == "matching":
        return (2 * both) / (len(detector_1) + len(detector_2))
    elif mode == "n_double":
        denom = len(detector_1) + len(detector_2) - both
        return both / denom if denom != 0 else 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}")

def compute_quality(data, fs):
    """
    Signal quality estimation between two R-peak detectors using physiological rules and bSQI.

    Parameters:
        data: List or tuple of two arrays [peaks1, peaks2] in sample indices
        fs: Sampling frequency in Hz

    Returns:
        Quality score in range [0, 100]
    """
    peaks1, peaks2 = data
    if len(peaks1) < 3 or len(peaks2) < 3:
        return 0

    # Convert R-peaks to milliseconds
    r_peaks_ms = (peaks1 / fs) * 1000
    rr_intervals_ms = np.diff(r_peaks_ms)
    rr_intervals_s = rr_intervals_ms / 1000

    # Heart rate calculation
    try:
        heart_rates = 60000 / rr_intervals_ms
        mean_hr = np.mean(heart_rates)
    except Exception:
        return 0

    # Thresholds
    HR_LOW = 40
    HR_HIGH = 180
    RR_MAX_S = 3
    RR_RATIO_LIMIT = 2.2

    # Rule 1: Physiological HR range
    if not (HR_LOW < mean_hr < HR_HIGH):
        return 0

    # Rule 2: No excessively long RR intervals
    if np.any(rr_intervals_s > RR_MAX_S):
        return 0

    # Rule 3: RR ratio stability
    rr_ratio = np.max(rr_intervals_s) / max(np.min(rr_intervals_s), 1e-6)
    if rr_ratio >= RR_RATIO_LIMIT:
        return 0

    # Rule 4: Template match (bSQI)
    try:
        match_score = bSQI(peaks1, peaks2, fs=fs)  # Returns 0 to 1
    except ZeroDivisionError:
        match_score = 0

    return round(match_score * 100, 2)


def segment_rrpeaks2(peaks,t,fs, wind= 10):
    segments = []
    for start in t:
        mask = (start < peaks/fs)  & (peaks/fs < start+wind)
        segment = peaks[mask]
        segments.append(segment)
    return segments

def quality(ecg_sen, fs):
    
    auto2 = ecg_sen.r_peaks
    ecg_temp1 = ECGlabeling("temp", ecg_sen.raw, freq=fs)

    ecg_temp1.auto_label(method="SSF")
    #ecg_temp1.auto_label(method="xqrs")

    auto1 = ecg_temp1.r_peaks


    t = np.arange(0, len(ecg_sen.raw) / fs, 1 /fs)
    t_pre1 = segment_rrpeaks2(auto2, np.arange(t[0], t[-1] - 10, 5), fs)
    t_pre2 = segment_rrpeaks2(auto1, np.arange(t[0], t[-1] - 10, 5), fs)

    qualitys = list()
    for split_d1, split_d2 in zip(t_pre1, t_pre2):
        signal_quality = compute_quality([split_d1, split_d2], fs)
        qualitys.append(signal_quality)
    qualitys = np.array(qualitys)
    #print(qualitys)
    print("mean qualtiy: ", np.median(qualitys))

    #qualitys[qualitys < 0] = 0
    return  qualitys, auto1
    

def add_quality_all(test,records,record_id,ecg_sen, qual):
    
    df = test.filter_table

    # Filter rows where 'column_name' is True
    df["quality"] = False
    
    # Assuming df is your DataFrame

    timestamps = np.array(ecg_sen.ts, dtype='datetime64[ns]')

    #ual, _ = quality(ecg_sen,ecg_sen.frequency)
    #new_timestamps = pd.date_range(start=timestamps[0], end=timestamps[-1], freq='5S')
    
    quality_start_time = timestamps[0] + np.timedelta64(5, 's')
    new_timestamps = pd.date_range(
        start=quality_start_time,
        periods=len(qual),
        freq='5S'
    )


    # Iterate over new_timestamps in chunks of 300 seconds
    for i in range(0, len(new_timestamps[:-60]), 1):  # 60 * 5 seconds = 300 seconds
        # Define the start and end indices of the current interval
        start_idx = i
        end_idx = min(i + 59, len(new_timestamps) - 1)

        # Segment the quality data for the current interval
        interval_quality = qual[start_idx:end_idx + 1]
        interval_area = np.mean(interval_quality)

        end_idx = min(i + 59, len(new_timestamps[:-60]) - 1)

        idx_from = test.timestamp_to_row(new_timestamps[start_idx])
        idx_to = test.timestamp_to_row(new_timestamps[end_idx])
        df.loc[idx_from:idx_to, "quality"] = interval_area
    print(test._filepath_filter)
    df.to_pickle(test._filepath_filter)