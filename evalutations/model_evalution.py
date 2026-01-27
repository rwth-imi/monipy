from __future__ import annotations

import os
import json
import warnings
from typing import Dict

import numpy as np
import pandas as pd

import optuna
import mlflow

from monipy.data.SeizureEvaluation import SeizureEvaluation
from monipy.models.tsai_models import ClassifierModel
import monipy.utils.detection_utils as detection_utils
from monipy.data.FeatureTable2 import FeatureTable

from reproducibility import set_global_seed

# ---------------------------------------------------------------------
# GLOBAL SETUP
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

set_global_seed(42, deterministic=True)

EVAL_WINDOW_SECONDS = 300
SECONDS_PER_DAY = 86400.0
QUALITY_THRESHOLD = 1

# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
class Manager:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        self.seizures_info = self._load_seizures_info()

    def _load_seizures_info(self) -> Dict[str, Dict]:
        path = os.path.join(self.base_dir, "seizures_details.json")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r") as f:
            return json.load(f)


def create_records(manager: Manager) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    for sid, info in manager.seizures_info.items():
        rid = info["record_start_sen"]
        if rid not in records:
            records[rid] = {
                "record_id": rid,
                "patient_id": info["patient_id"],
                "record_start": np.datetime64(info["record_start_sen"]),
                "record_end": np.datetime64(info["record_end_sen"]),
                "seizures": [],
            }
        records[rid]["seizures"].append(sid)
    return records


def build_grouped_from_base_dir(base_dir: str) -> Dict:
    manager = Manager(base_dir)
    records = create_records(manager)

    grouped: Dict = {}
    for rid, rec in records.items():
        pid = rec["patient_id"]
        grouped.setdefault(pid, {})
        grouped[pid][rid] = rec
    return grouped

# ---------------------------------------------------------------------
# DEFICIENCY
# ---------------------------------------------------------------------
def compute_deficiency(feature_table: FeatureTable, quality_threshold: int) -> np.timedelta64:
    bad = feature_table.filter_table["quality"] < quality_threshold
    if not bad.any():
        return np.timedelta64(0, "s")

    width = feature_table.ALL_FEATURES_WINDOW_WIDTH
    idx = np.where(bad.to_numpy())[0]

    total = np.timedelta64(0, "s")
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i != prev + 1:
            t0 = feature_table.row_to_timestamp(start)
            t1 = feature_table.row_to_timestamp(prev) + np.timedelta64(width, "s")
            total += (t1 - t0)
            start = i
        prev = i

    t0 = feature_table.row_to_timestamp(start)
    t1 = feature_table.row_to_timestamp(prev) + np.timedelta64(width, "s")
    total += (t1 - t0)

    return total

# ---------------------------------------------------------------------
# SEIZURE + DETECTION EVALUATION
# ---------------------------------------------------------------------
def evaluate_detections_on_record(record_id, detections, info_df, onset_col, offset_col):
    seizures = []
    rec_df = info_df[info_df["record_id"] == record_id]

    for _, row in rec_df.iterrows():
        s = SeizureEvaluation(
            record_id,
            row["seizure_id"],
            np.datetime64(row[onset_col]),
            np.datetime64(row[offset_col]),
            {},
        )
        s.patient_id = row["patient_id"]
        s.is_detected = False
        seizures.append(s)

    for d in detections:
        d.set_status("fp")

    for s in seizures:
        for d in detections:
            if abs(d.timestamp_start - s.timestamp_start) <= np.timedelta64(
                EVAL_WINDOW_SECONDS, "s"
            ):
                s.is_detected = True
                d.set_status("tp")

    return seizures, detections

# ---------------------------------------------------------------------
# TABLE BUILDERS
# ---------------------------------------------------------------------
def build_sec_evalu(seizures):
    return pd.DataFrame([{
        "patient_id": s.patient_id,
        "record_id": s.record_id,
        "seizure_id": s.seizure_id,
        "is_detected": s.is_detected,
    } for s in seizures])


def build_detection_results(detections, patient_id, record_id):
    return pd.DataFrame([{
        "patient_id": patient_id,
        "record_id": record_id,
        "result": d.get_status(),
    } for d in detections])


def build_first_evalu(sec_df, det_df, record_start, record_end):
    ft = FeatureTable(record_start, record_end)
    deficiency = compute_deficiency(ft, QUALITY_THRESHOLD)

    duration = record_end - record_start
    fp = (det_df["result"] == "fp").sum()
    tp = (det_df["result"] == "tp").sum()

    return pd.DataFrame([{
        "patient_id": sec_df.patient_id.iloc[0],
        "record_id": sec_df.record_id.iloc[0],
        "record_duration": duration,
        "deficiency": deficiency,
        "n_det_fp": fp,
        "n_det_tp": tp,
    }])

# ---------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------
def evaluate_model_on_data_package(
    model,
    data,
    labels,
    meta_df,
    grouped,
    threshold,
    onset_col,
    offset_col,
):
    preds = model.predict(data)

    # Window-level detections reused per record (intentional simplification)
    detections = detection_utils.get_detections(preds, threshold)

    all_sec, all_first, all_det = [], [], []

    for pid, records in grouped.items():
        for rid, rec in records.items():
            seizures, dets = evaluate_detections_on_record(
                rid, detections, meta_df, onset_col, offset_col
            )

            sec = build_sec_evalu(seizures)
            det = build_detection_results(dets, pid, rid)
            first = build_first_evalu(
                sec, det, rec["record_start"], rec["record_end"]
            )

            all_sec.append(sec)
            all_det.append(det)
            all_first.append(first)

    return (
        pd.concat(all_sec, ignore_index=True),
        pd.concat(all_first, ignore_index=True),
        pd.concat(all_det, ignore_index=True),
    )

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def compute_patient_metrics(pid, records_df, seizures_df):
    rec = records_df[records_df.patient_id == pid]
    sei = seizures_df[seizures_df.patient_id == pid]

    duration_days = rec.record_duration.sum() / np.timedelta64(1, "D")
    tp = int(sei.is_detected.sum())
    fn = int((~sei.is_detected).sum())
    fp = int(rec.n_det_fp.sum())

    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * ppv * sen / (ppv + sen) if (ppv + sen) else 0.0
    far = fp / duration_days if duration_days else 0.0

    return tp, fp, fn, ppv, sen, f1, far


def compute_cohort_metrics(seizures, records):
    rows = []
    for pid in records.patient_id.unique():
        rows.append((pid, *compute_patient_metrics(pid, records, seizures)))

    df = pd.DataFrame(rows, columns=[
        "patient_id","TP","FP","FN","PPV","Sensitivity","F1","FAR"
    ])

    return {
        "Total_Sensitivity": df.TP.sum() / (df.TP.sum() + df.FN.sum()) * 100,
        "Total_FAR": df.FP.sum() / (records.record_duration.sum() / np.timedelta64(1, "D")),
        "med_F1": df.F1.median(),
        "med_FAR": df.FAR.median(),
        "med_FP": df.FP.median(),
        "med_PPV": df.PPV.median(),
        "N_tp": df.TP.sum(),
        "N_fp": df.FP.sum(),
        "N_fn": df.FN.sum(),
        "n_seizures": len(seizures),
    }

# ---------------------------------------------------------------------
# RUNNER
# ---------------------------------------------------------------------
def run_experiment(data, labels, meta_df, base_dir):
    mlflow.set_experiment("simplified_experiment")
    grouped = build_grouped_from_base_dir(base_dir)

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(
        lambda t: objective(t, data, labels, meta_df, grouped),
        n_trials=20,
    )
    return study
