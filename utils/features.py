import numpy as np

from typing import Tuple, Dict, List
import scipy.stats
from scipy.signal import detrend
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import fftpack
from MFDFA import MFDFA


feature_names = [
    "avg",
    "sd",
    "rmssd",
    "rmssd_dt",
    "skew",
    "kurt",
    "pnnx",
    "nnx",
    "triangular_index",
    "quantile_25",
    "quantile_50",
    "quantile_75",
    "variance",
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


def feature_to_index(feature: str) -> int:
    if feature in feature_names:
        return feature_names.index(feature)
    else:
        raise RuntimeError(
            f"{feature} is not a valid feature name. Correct names are: {feature_names}"
        )


def index_to_feature(index: int) -> str:
    return feature_names[index]


# ========================================================================
# Simple Features
# ========================================================================


def avg(rri: np.ndarray) -> np.ndarray:
    return np.mean(rri)


def sd(rri: np.ndarray) -> np.ndarray:
    return np.std(rri)


def rmssd(rri: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.diff(rri))))


def rmssd_dt(rri: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(np.diff(detrend(rri)))))


def skew(rri: np.ndarray) -> np.ndarray:
    return scipy.stats.skew(rri)


def kurt(rri: np.ndarray) -> np.ndarray:
    return scipy.stats.kurtosis(rri)


def pnnx(rri: np.ndarray, alpha: int = 50) -> np.ndarray:
    diff = np.diff(rri)
    return nnx(rri, alpha) / len(diff)


def nnx(rri: np.ndarray, alpha: int = 50) -> np.ndarray:
    diff = np.diff(rri)
    return np.sum(np.abs(diff) >= alpha)


def triangular_index(rri: np.ndarray) -> np.ndarray:
    edges = np.arange(0, 2500, 1 / 128 * 1000)
    histcounts, _ = np.histogram(rri, edges)
    return np.sum(histcounts) / np.max(histcounts)


def quantile_25(rri: np.ndarray) -> np.ndarray:
    return np.quantile(rri, 0.25)


def quantile_50(rri: np.ndarray) -> np.ndarray:
    return np.quantile(rri, 0.50)


def quantile_75(rri: np.ndarray) -> np.ndarray:
    return np.quantile(rri, 0.75)


def variance(rri: np.ndarray) -> np.ndarray:
    return np.var(rri)


# ========================================================================
# Poincare Features
# ========================================================================


def poincare(rri: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute poincare features from rri

    Args:
        rri (np.ndarray): RRI data.

    Returns:
        result (Dict[str, np.ndarray]): Feature vectors.

    """
    sd_diff = np.std(np.diff(rri))
    sd_rr = np.std(rri)
    result = dict()

    result["sd1"] = (1 / np.sqrt(2)) * sd_diff
    result["sd2"] = np.sqrt((2 * sd_rr ** 2) - (0.5 * sd_diff ** 2))

    t = 4 * result["sd1"]
    l = 4 * result["sd2"]

    result["csi"] = l / t
    result["csim"] = np.square(l) / t
    result["cvi"] = np.log10(l * t)

    return result


def poincare_filtered_slope(
    rri: np.ndarray, poincare_features: Dict[str, np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute poincare features (including ones using slopes) from rri

    Args:
        rri (np.ndarray): RRI data.
        poincare_features (Dict[str, np.ndarray]): Poincare (without slopes) feature vectors

    Returns:
        result (Dict[str, np.ndarray]): Feature vectors.

    """
    slope = comp_slope(rri)

    if poincare_features is None:
        poincare_features = poincare(rri)

    result = dict()
    result["csi_slope"] = slope * poincare_features["csi"]
    result["csim_slope"] = slope * poincare_features["csim"]

    poincare_features_filtered = poincare(comp_median_filter(rri))
    result["csi_filtered"] = poincare_features_filtered["csi"]
    result["csim_filtered"] = poincare_features_filtered["csim"]

    result["csi_filtered_slope"] = slope * poincare_features_filtered["csi"]
    result["csim_filtered_slope"] = slope * poincare_features_filtered["csim"]

    return result


# ========================================================================
# HR Diff Features
# ========================================================================


def hr_diff_all(rri: np.ndarray) -> np.ndarray:
    slope = comp_slope(rri)
    hr_diff = comp_hr_diff(rri)
    hr_diff_filt = comp_hr_diff(comp_median_filter(rri))

    result = dict()
    result["hr_diff"] = hr_diff
    result["hr_diff_slope"] = slope * hr_diff
    result["hr_diff_filtered"] = hr_diff_filt
    result["hr_diff_filtered_slope"] = slope * hr_diff_filt

    return result


# ========================================================================
# Frequency Features
# ========================================================================


def freq_based(rri: np.ndarray, resampling_rate: int = 7) -> Dict[str, np.ndarray]:
    """
    Computes frequency features.

    Args:
        rri (np.ndarray): RRI data.
        resampling_rate (int, optional): Resampling rate ind Hertz. Defaults to 7.

    Returns:
        result (Dict[str, np.ndarray]): Feature vectors.

    """
    try:
        # resampling rate In Hertz
        total_sec_rr = np.cumsum(np.divide(rri, 1000)) - rri[0] / 1000

        sample_indizes = np.arange(
            total_sec_rr[0], total_sec_rr[-1], 1 / resampling_rate
        )
        sample_func = interp1d(total_sec_rr, rri, "cubic")  # Returns function

        resampled_rr = np.round(sample_func(sample_indizes) / 1000, 4)

        nfft = np.power(2, np.ceil(np.log2(np.abs(len(resampled_rr)))))
        fft_rr = fftpack.fft(scipy.stats.zscore(resampled_rr), int(nfft)) / len(
            resampled_rr
        )
        frequencies = resampling_rate / 2 * np.linspace(0, 1, int(nfft / 2) + 1)

        psd_rr = 2 * np.abs(fft_rr[0 : int(nfft / 2) + 2])

        freq_limits = {
            "ulf": [0, 0.003],
            "vlf": [0.003, 0.04],
            "lf": [0.04, 0.15],
            "hf": [0.15, 0.4],
        }

        result = dict()
        freq_limits_idc = dict()
        for key in freq_limits:
            freq_limits_idc.update(
                {
                    key: np.argwhere(
                        (frequencies >= freq_limits[key][0])
                        & (frequencies <= freq_limits[key][1])
                    ).flatten()
                }
            )

            result[key] = np.sum(psd_rr[freq_limits_idc[key]])

        result["total_power"] = np.sum([result[key] for key in freq_limits.keys()])

        lf_normalized = result["lf"] / result["total_power"]
        hf_normalized = result["hf"] / result["total_power"]
        result["lf_hf_ratio"] = np.round(lf_normalized / hf_normalized * 100) / 100

    except Exception as e:
        print(f"Error on computing freq_based: {str(e)}")
        print("Data was: ")
        print(rri)
        result = dict()
        for key in ["ulf", "vlf", "lf", "hf", "total_power", "lf_hf_ratio"]:
            result[key] = np.NaN

    return result


# ========================================================================
# Multifractal features
# ========================================================================
def get_multifractal_spectrum_trends(
    rri: np.ndarray,
) -> Tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Calculates the multifractal spectrum (hurst exponent vs q-power). The spectrum is divided in 3 equally sized
    segments (left, center right). For each segment of the spectrum the linear trends are computed. The trends are
    returned, along with the highest value of the spectrum. If the calculation of the spectrum fails, the zero values
    are returned.
    Args:
        rri: an rri segment

    Returns:
        spec_max: the highest value of the spectrum
        coef_left: the left side trend of the spectrum
        coef_center: the center trend of the spectrum
        coef_right the left side trend of the spectrum

    """
    try:
        lag, dfa, q_list, hurst = get_hurst_exponents(rri)

        coef_left, coef_center, coef_right = fit_piece_wise_linear(q_list, hurst)

        return hurst.max(), coef_left, coef_center, coef_right
    except Exception as e:
        print(e)
        return 0, 0, 0, 0


def get_hurst_exponents(
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The function computing the multifractal hurst exponents.
    Args:
        x:

    Returns:
        lag: Array of lags, realigned, preserving only different lags and with entries > order + 1
        dfa: Array of variances of the detrended fluctuations at different lags and q-fractal powers.
        q_list: Array of values representing fractal powers upon which the fluctuations are decomposed.
        hurst: the hurst exponents derived from the linear fitting of dfa vs q

    """
    X = x[~np.isnan(x)]
    q_list = np.linspace(-10, 10, 20)
    q_list = q_list[q_list != 0.0]

    lag = np.linspace(1, (len(X) - 1) / 4, 10).astype(int)

    # The order of the polynomial fitting
    order = 1
    # Obtain the (MF)DFA as
    lag, dfa = MFDFA(X, lag=lag, q=q_list, order=order)
    final_lag, final_dfa = [], []
    for i, d in enumerate(dfa):
        if 0 in d:
            continue
        final_lag.append(lag[i])
        final_dfa.append(d)

    hurst = np.polyfit(np.log(final_lag), np.log(final_dfa), 1)[0]

    return lag[4:], dfa[4:], q_list[4:], hurst[4:]


# ========================================================================
# Helpers
# ========================================================================


def comp_hr_diff(rri: np.ndarray) -> np.ndarray:
    hr_diff = 0
    for i in range(1, len(rri) - 1):
        hr_diff += 0.5 * (rri[i + 1] - rri[i - 1])
    return hr_diff


def comp_slope(rri: np.ndarray) -> float:
    rri_filt = comp_median_filter(rri)

    def f(x, a, b):
        return a * x + b  # Straight line

    f_params, _ = curve_fit(f, range(len(rri_filt)), rri_filt)

    return f_params[0]  # Slope of computed straight


# todo: is this the best way to do it?
def comp_median_filter(rri: np.ndarray, window_length: int = 7) -> List[float]:
    result = []

    for i in range(window_length, len(rri) + 1):
        result.append(np.median(rri[i - window_length : i]))

    return result


def fit_piece_wise_linear(
    x_data: np.ndarray, y_data: np.ndarray
) -> Tuple[np.float64, np.float64, np.float64]:
    idx_left = np.array(range(int(len(x_data) / 3)))
    idx_center = np.array(range(int(len(x_data) / 3), int(len(x_data) * 2 / 3)))
    idx_right = np.array(range(int(len(x_data) * 2 / 3), int(len(x_data))))

    coef_left = np.polyfit(x_data[idx_left], y_data[idx_left], 1)[0]
    coef_center = np.polyfit(x_data[idx_center], y_data[idx_center], 1)[0]
    coef_right = np.polyfit(x_data[idx_right], y_data[idx_right], 1)[0]

    return coef_left, coef_center, coef_right
