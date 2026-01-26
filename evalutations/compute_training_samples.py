
import json
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

import monipy.data.tools as datatools

from monipy.data.FeatureTable2 import FeatureTable as ft
from monipy.data.Record2 import Record2
from monipy.data.RRISection import RRISection
from ECGlabeling import ECGlabeling

# =============================================================================

def load_seizures_info(data_root: str) -> dict:
    with open(os.path.join(data_root, "seizures_details.json"), "r") as f:
        return json.load(f)


def load_ecg(data_root: str, relative_path: str) -> ECGlabeling:
    ecg = ECGlabeling()
    ecg.load_data(os.path.join(data_root, relative_path))
    return ecg


def compute_rr(ecg: ECGlabeling):
    if len(ecg.r_peaks) < 2:
        return np.array([]), pd.DatetimeIndex([])

    r_sec = ecg.r_peaks / ecg.frequency
    rr_ms = np.diff(r_sec) * 1000
    start = pd.to_datetime(ecg.ts[0])
    ts = start + pd.to_timedelta(r_sec[1:], unit="s")
    return rr_ms, ts


# =============================================================================

def create_records(seizures_info: dict) -> dict:
    records = {}
    seen = set()

    for idx, (sid, s) in enumerate(seizures_info.items()):
        start = s["record_start_sen"]
        if start in seen:
            continue

        seen.add(start)
        records[f"record_{idx}"] = {
            "record_start_sen": s["record_start_sen"],
            "record_end_sen": s["record_end_sen"],
            "patient_id": s["patient_id"],
            "path_sensor": s["path_sensor"],
            "seizures": [],
            "unmasked": None,
        }

    for sid, s in seizures_info.items():
        for rec in records.values():
            if rec["record_start_sen"] == s["record_start_sen"]:
                rec["seizures"].append(sid)

    for rec in records.values():
        if rec["seizures"]:
            rec["unmasked"] = os.path.join(rec["seizures"][-1], "unmasked")

    return records


# =============================================================================

def get_seizure_times(seizure_id, info_df, onset_col, offset_col):
    row = info_df[info_df["seizure_id"] == int(seizure_id)]
    if row.empty:
        return None, None
    return (
        np.datetime64(row[onset_col].values[0]),
        np.datetime64(row[offset_col].values[0]),
    )


def get_seizures(info_df, records, record_id, onset, offset):
    rows = []

    for seizure_id in records[record_id]["seizures"]:
        start_ts, end_ts = get_seizure_times(
            seizure_id, info_df, onset, offset
        )
        rows.append(
            {
                "record_id": record_id,
                "seizure_id": seizure_id,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "has_valid_data": True,
                "is_event": True,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================

def sample_seizure_data(
    records,
    data_root,
    info_df,
    onset,
    event_filter_id,
    quality,
    sample_windows,
):
    seizure_data = []
    metadata = datatools.get_empty_metadata_dict()

    for record_id in tqdm(records.keys(), desc="Sampling seizures"):
        record = records[record_id]

        feature_table = ft(
            records,
            record_id,
            None,
            onset,
            info=info_df,
        )

        header = {
            "timestamp_start": np.datetime64(record["record_start_sen"]),
            "timestamp_end": np.datetime64(record["record_end_sen"]),
        }

        feature_table.record = Record2(
            file_name=None,
            header=header,
            device_specific_header=None,
            rpeak_indices=None,
            rpeak_classes=None,
            rpeak_timestamps=np.array([0]),
            intervals=RRISection(np.array([0]), np.array([0]), np.array([0])),
        )

        for seizure_id in record["seizures"]:
            start_ts, _ = get_seizure_times(
                seizure_id, info_df, onset, onset.replace("onset", "offset")
            )
            if start_ts is None:
                continue

            features = feature_table.get_sampling_window_features(
                sampling_time=start_ts,
                sampling_windows=sample_windows,
                event_filter_id=event_filter_id,
                quality=quality,
            )

            for item in features:
                seizure_data.append(item["feature_data"].to_numpy())
                metadata["timestamp_start"].append(item["timestamp_start"])
                metadata["seizure_id"].append(seizure_id)
                metadata["record_id"].append(record_id)
                metadata["window"].append(item["window_index"])

    labels = np.ones(len(seizure_data), dtype=int)
    return np.array(seizure_data), metadata, labels


# =============================================================================

def sample_regular_data(
    records,
    info_df,
    onset,
    event_filter_id,
    quality,
    sample_windows,
    n_samples,
    seed=42,
):
    np.random.seed(seed)
    regular_data = []
    metadata = datatools.get_empty_metadata_dict()

    for record_id in tqdm(records.keys(), desc="Sampling regular"):
        record = records[record_id]

        feature_table = ft(
            records,
            record_id,
            None,
            onset,
            info=info_df,
        )

        feature_table.check_for_event_filter(event_filter_id)

        candidates = feature_table.filter_table[
            (feature_table.filter_table["without_seizure"]) &
            (feature_table.filter_table["quality"] >= quality)
        ].sample(frac=1)

        for idx in candidates.index:
            ts = feature_table.row_to_timestamp(idx)

            features = feature_table.get_sampling_window_features(
                sampling_time=ts,
                sampling_windows=sample_windows,
                event_filter_id=event_filter_id,
                quality=quality,
            )

            for item in features:
                regular_data.append(item["feature_data"].to_numpy())
                metadata["timestamp_start"].append(item["timestamp_start"])
                metadata["seizure_id"].append(np.nan)
                metadata["record_id"].append(record_id)
                metadata["window"].append(item["window_index"])

                if len(regular_data) >= n_samples:
                    return np.array(regular_data), metadata

    return np.array(regular_data), metadata


# =============================================================================

def collect_data(
    data_root,
    info_df,
    onset,
    event_filter_id,
    quality,
    class_ratio,
    sample_windows,
):
    seizures_info = load_seizures_info(data_root)
    records = create_records(seizures_info)

    seizure_data, seizure_meta, seizure_labels = sample_seizure_data(
        records,
        data_root,
        info_df,
        onset,
        event_filter_id,
        quality,
        sample_windows,
    )

    regular_data, regular_meta = sample_regular_data(
        records,
        info_df,
        onset,
        event_filter_id,
        quality,
        sample_windows,
        n_samples=len(seizure_data) * class_ratio,
    )

    data = np.vstack([seizure_data, regular_data]).astype(np.float32)
    labels = np.concatenate([seizure_labels, np.zeros(len(regular_data))])

    meta = pd.concat(
        [pd.DataFrame(seizure_meta), pd.DataFrame(regular_meta)],
        ignore_index=True,
    )

    meta["label"] = labels
    meta["patient_id"] = meta["record_id"].map(
        lambda x: records[x]["patient_id"]
    )

    return data, meta, labels
