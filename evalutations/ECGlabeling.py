import math
from typing import Optional

import numpy as np
from tqdm import tqdm
from scipy.io import savemat, loadmat

from biosppy.signals import ecg as biosppy_ecg
from ecgdetectors import Detectors
import neurokit2 as nk
from wfdb import processing


# =============================================================================
# Helper
# =============================================================================

def refine_peaks(signal: np.ndarray, candidate_peaks: np.ndarray) -> np.ndarray:
    refined = []

    for peak in candidate_peaks:
        left = peak
        while left > 0 and signal[left - 1] >= signal[left]:
            left -= 1

        right = peak
        while right < len(signal) - 1 and signal[right + 1] >= signal[right]:
            right += 1

        refined.append(left if signal[left] > signal[right] else right)

    return np.array(refined, dtype=int)


# =============================================================================
# ECG labeling
# =============================================================================

class ECGlabeling:
    def __init__(
        self,
        name: str = "",
        raw: Optional[np.ndarray] = None,
        freq: int = 0,
        offset: int = 0,
        label: Optional[str] = None,
        ts: Optional[np.ndarray] = None,
        info: Optional[dict] = None,
    ):
        self.name = name
        self.raw = raw if raw is not None else np.array([])
        self.frequency = freq
        self.r_peaks = np.array([], dtype=int)
        self.offset = offset
        self.label = label
        self.ts = ts if ts is not None else np.array([])
        self.info = info

    # -------------------------------------------------------------------------
    # Automatic R-peak detection
    # -------------------------------------------------------------------------

    def auto_label(
        self,
        method: str = "Engzee",
        segment_length: Optional[int] = None,
    ) -> None:
        self.r_peaks = np.array([], dtype=int)

        n_segments = math.ceil(len(self.raw) / segment_length)

        for i in tqdm(range(n_segments), desc="Detecting R-peaks"):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(self.raw))
            segment = self.raw[start:end]

            peaks = self._detect_peaks(segment, method)
            if peaks.size == 0:
                continue

            peaks = refine_peaks(segment, peaks)
            self.r_peaks = np.concatenate(
                (self.r_peaks, peaks + start),
                axis=0,
            )

        self.r_peaks = np.unique(self.r_peaks)

    # -------------------------------------------------------------------------
    # Peak detectors
    # -------------------------------------------------------------------------

    def _detect_peaks(self, signal: np.ndarray, method: str) -> np.ndarray:
        if method == "Engzee":
            return biosppy_ecg.engzee_segmenter(
                signal=signal,
                sampling_rate=self.frequency,
            )[0]

        if method == "Hamilton":
            try:
                return biosppy_ecg.ecg(
                    signal=signal,
                    sampling_rate=self.frequency,
                    show=False,
                )[2]
            except ValueError:
                return np.array([], dtype=int)

        if method == "SSF":
            return biosppy_ecg.ssf_segmenter(
                signal=signal,
                sampling_rate=self.frequency,
                threshold=0.00005,
                before=0.35,
                after=0.35,
            )[0]

        if method == "WQRS":
            return np.array(
                Detectors(self.frequency).wqrs_detector(signal),
                dtype=int,
            )

        if method.startswith("Neurokit_"):
            nk_method = method.replace("Neurokit_", "")
            _, info = nk.ecg_process(
                signal,
                sampling_rate=self.frequency,
                method=nk_method,
            )
            return np.where(info["ECG_R_Peaks"] == 1)[0]

        if method == "xqrs":
            detector = processing.XQRS(sig=signal, fs=self.frequency)
            detector.detect()
            return detector.qrs_inds

        raise ValueError(
            "Unknown method. Use Engzee, Hamilton, SSF, WQRS, xqrs, or Neurokit_*"
        )

    # -------------------------------------------------------------------------
    # IO
    # -------------------------------------------------------------------------

    def save_data(self) -> None:
        data = {
            "raw_data": self.raw,
            "frequency": self.frequency,
            "r_peaks": self.r_peaks,
            "offset": self.offset,
            "ts": self.ts,
        }

        try:
            savemat(f"{self.name}.mat", data)
        except Exception:
            savemat(f"{self.name}.mat", data, format="4")

    def load_data(self, file_name: str) -> None:
        mat = loadmat(f"{file_name}.mat")
        self.name = file_name
        self.raw = mat["raw_data"].ravel()
        self.frequency = int(mat["frequency"][0][0])
        self.r_peaks = mat["r_peaks"].ravel().astype(int)
        self.offset = int(self.r_peaks[0]) if len(self.r_peaks) > 0 else 0
        self.ts = mat["ts"].ravel()
