from __future__ import annotations

import os
import json
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch

import optuna
import mlflow

from ECGlabeling import ECGlabeling

from monipy.data.FeatureTable2 import FeatureTable as ft
from monipy.data.SeizureEvaluation import SeizureEvaluation
from monipy.data.Detection import Detection
from monipy.models.tsai_models import ClassifierModel

import monipy.utils.detection_utils as detection_utils

from reproducibility import set_global_seed


warnings.filterwarnings("ignore", category=RuntimeWarning)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

set_global_seed(42, deterministic=True)



EVAL_WINDOW_SECONDS: int = 300
SECONDS_PER_DAY: float = 86400.0

FEATURE_NAMES: List[str] = [
    "avg","sd","rmssd","rmssd_dt","skew","kurt","pnnx","nnx","triangular_index",
    "quantile_25","quantile_50","quantile_75","variance","csi","csim","cvi",
    "sd1","sd2","csi_slope","csim_slope","csi_filtered","csim_filtered",
    "csi_filtered_slope","csim_filtered_slope","hr_diff","hr_diff_slope",
    "hr_diff_filtered","hr_diff_filtered_slope"
]

def feature_to_index(name: str) -> int:
    return FEATURE_NAMES.index(name)


class Manager:
    def __init__(self, base_dir: str) -> None:
        self.base_dir: str = base_dir
        self.seizures_info: Dict[str, Dict] = self._load_seizures_info()
        self.seizure_ids: List[str] = list(self.seizures_info.keys())

    def _load_seizures_info(self) -> Dict[str, Dict]:
        path = os.path.join(self.base_dir, "seizures_details.json")
        with open(path, "r") as f:
            return json.load(f)

    def load_ecg(self, path: str) -> ECGlabeling:
        ecg = ECGlabeling()
        ecg.load_data(path)
        return ecg

    def get_rr(
        self, ecg: ECGlabeling
    ) -> Tuple[np.ndarray, np.ndarray]:
        ts = pd.to_datetime(ecg.ts)
        r_peaks_s = ecg.r_peaks / ecg.frequency
        r_times = ts[0] + pd.to_timedelta(r_peaks_s, unit="s")
        return np.diff(r_peaks_s) * 1000, r_times[1:]


def create_records(manager: Manager) -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    seen: set[str] = set()

    for sid, info in manager.seizures_info.items():
        start = info["record_start_sen"]
        if start in seen:
            continue
        seen.add(start)

        records[start] = {
            "record_start": np.datetime64(info["record_start_sen"]),
            "record_end": np.datetime64(info["record_end_sen"]),
            "patient_id": info["patient_id"],
            "seizures": []
        }

    for sid, info in manager.seizures_info.items():
        records[info["record_start_sen"]]["seizures"].append(sid)

    return records


def get_pat_sei_stats(
    seizure_df: pd.DataFrame,
    patient_id: str | int
) -> Tuple[int, int]:
    df = seizure_df[seizure_df["patient_id"] == patient_id]
    counts = df["is_detected"].value_counts()
    tp = int(counts.get(True, 0))
    fn = int(counts.get(False, 0))
    return tp, fn

def compute_patient_metrics(
    patient_id: str | int,
    detection_df: pd.DataFrame,
    seizure_df: pd.DataFrame
) -> Tuple[float, int, int, int, int, float, float, float, float]:
    det = detection_df[detection_df["patient_id"] == patient_id]
    sei = seizure_df[seizure_df["patient_id"] == patient_id]

    total_seconds = det["record_duration"].sum() / np.timedelta64(1, "s")
    duration_hours = total_seconds / 3600.0
    duration_days = total_seconds / SECONDS_PER_DAY

    n_seizures = len(sei)
    tp, fn = get_pat_sei_stats(seizure_df, patient_id)
    fp = int(det["fp"].sum())

    far24 = fp / duration_days if duration_days > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * ppv * sen / (ppv + sen) if (ppv + sen) > 0 else 0.0

    return duration_hours, n_seizures, tp, fp, fn, far24, ppv, sen, f1


def evaluate_detections_on_record(
    record_id: str,
    detections: List[Detection],
    info_df: pd.DataFrame,
    onset_col: str,
    offset_col: str
) -> Tuple[List[SeizureEvaluation], List[Detection]]:
    seizures: List[SeizureEvaluation] = []

    rec_df = info_df[info_df["record_id"] == record_id]

    for _, row in rec_df.iterrows():
        seizure = SeizureEvaluation(
            record_id,
            row["seizure_id"],
            np.datetime64(row[onset_col]),
            np.datetime64(row[offset_col]),
            {},
        )
        seizure.patient_id = row["patient_id"]
        seizure.is_detected = False
        seizures.append(seizure)

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


def evaluate_model_on_data_package(
    model: ClassifierModel,
    data: np.ndarray,
    labels: np.ndarray,
    meta_df: pd.DataFrame,
    grouped_records: Dict,
    threshold: float,
    onset_col: str,
    offset_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seizure_rows: List[Dict] = []
    detection_rows: List[Dict] = []

    preds = model.predict(data)

    detections = detection_utils.get_detections(
        predictions=preds,
        threshold=threshold
    )

    for patient_id, records in grouped_records.items():
        for record_id in records.keys():
            seizures, dets = evaluate_detections_on_record(
                record_id,
                detections,
                meta_df,
                onset_col,
                offset_col,
            )

            for s in seizures:
                seizure_rows.append({
                    "patient_id": patient_id,
                    "record_id": record_id,
                    "seizure_id": s.seizure_id,
                    "is_detected": s.is_detected,
                })

            fp = sum(d.get_status() == "fp" for d in dets)
            detection_rows.append({
                "patient_id": patient_id,
                "record_id": record_id,
                "fp": fp,
                "record_duration": np.timedelta64(1, "D"),
            })

    return pd.DataFrame(seizure_rows), pd.DataFrame(detection_rows)

def model_performance(
    seizure_df: pd.DataFrame,
    detection_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for pid in seizure_df["patient_id"].unique():
        rows.append((pid, *compute_patient_metrics(pid, detection_df, seizure_df)))

    return pd.DataFrame(
        rows,
        columns=[
            "patient_id","duration_h","n_seizures",
            "tp","fp","fn","far24","ppv","sen","f1",
        ],
    )


def objective(
    trial: optuna.Trial,
    data: np.ndarray,
    labels: np.ndarray,
    meta_df: pd.DataFrame,
    grouped: Dict,
) -> Tuple[float, float]:
    config = {
        "model_name": "TransformerClassifier",
        "input_shape": (data.shape[1], data.shape[2]),
        "nb_classes": 1,
        "epochs": trial.suggest_int("epochs", 10, 30),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "prediction_threshold": trial.suggest_float("threshold", 0.3, 0.9),
    }

    model = ClassifierModel(config)

    model.scaler.fit(data)
    x = model.scaler.transform(data)
    y = labels

    model._train_core(x, y, None, None, n_fold=-1, verbose=0)

    sei_df, det_df = evaluate_model_on_data_package(
        model,
        x,
        y,
        meta_df,
        grouped,
        threshold=config["prediction_threshold"],
        onset_col="onset",
        offset_col="offset",
    )

    perf = model_performance(sei_df, det_df)
    mean_sen = perf["sen"].mean()
    mean_far = perf["far24"].mean()

    mlflow.log_params(config)
    mlflow.log_metric("mean_sensitivity", mean_sen)
    mlflow.log_metric("mean_far24", mean_far)

    return mean_sen, mean_far



def run_experiment(
    data: np.ndarray,
    labels: np.ndarray,
    meta_df: pd.DataFrame,
    grouped: Dict
) -> optuna.Study:
    mlflow.set_experiment("exp1")

    study = optuna.create_study(
        directions=["maximize", "minimize"]
    )

    study.optimize(
        lambda t: objective(t, data, labels, meta_df, grouped),
        n_trials=20,
    )

    return study
