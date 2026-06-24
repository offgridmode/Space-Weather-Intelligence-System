import numpy as np
import pandas as pd
import pywt
from src.config import (
    DETECTION_WAVELET, DETECTION_THRESHOLD_SIGMA,
    DETECTION_MIN_EVENT_SECONDS, DETECTION_MERGE_GAP_SECONDS,
)


def compute_baseline(flux: np.ndarray, window: int = 3600) -> np.ndarray:
    s = pd.Series(flux)
    med = s.rolling(window, center=True, min_periods=1).median().bfill().ffill()
    return med.values.astype(flux.dtype)


def cwt_detect(flux: np.ndarray, times: pd.DatetimeIndex,
               wavelet: str = "morl", max_scale: int = 64,
               threshold_sigma: float = DETECTION_THRESHOLD_SIGMA) -> pd.DataFrame:
    scales = np.arange(1, max_scale + 1)
    coeffs, _ = pywt.cwt(flux, scales, wavelet, sampling_period=1)
    power = np.abs(coeffs) ** 2
    avg_power = np.mean(power, axis=0)
    baseline = pd.Series(avg_power).rolling(600, center=True, min_periods=1).median()
    baseline = baseline.bfill().ffill().values
    excess = avg_power - baseline
    excess = np.maximum(excess, 0)
    valid = excess[~np.isnan(excess)]
    if len(valid) < 10:
        return _detect_by_excess(flux, times, threshold_sigma)
    noise_std = np.nanstd(valid[valid < np.nanpercentile(valid, 50)])
    if noise_std < 1e-12:
        noise_std = np.nanstd(valid) * 0.5
    threshold = threshold_sigma * max(noise_std, 1e-12)
    is_flare = excess > threshold
    return _extract_events(is_flare, times, flux, excess, avg_power)


def _estimate_noise(flux: np.ndarray) -> float:
    """Robust noise estimation using median absolute deviation."""
    finite = flux[np.isfinite(flux)]
    if len(finite) < 10:
        return 1.0
    mad = np.median(np.abs(np.diff(finite)))
    sigma = mad / 0.6745 if mad > 0 else np.std(finite) * 0.5
    return max(sigma, 1e-12)


def _detect_by_excess(flux: np.ndarray, times: pd.DatetimeIndex,
                      threshold_sigma: float) -> pd.DataFrame:
    baseline = compute_baseline(flux)
    excess = flux - baseline
    excess = np.maximum(excess, 0)
    if np.all(excess == 0):
        return pd.DataFrame()
    noise_std = _estimate_noise(excess)
    threshold = threshold_sigma * noise_std
    above = excess > threshold
    if not above.any():
        threshold = max(threshold_sigma * np.nanstd(excess) * 0.5, 1e-12)
        above = excess > threshold
    return _extract_events(above, times, flux, excess, excess)


def _extract_events(is_flare: np.ndarray, times: pd.DatetimeIndex,
                    flux: np.ndarray, excess: np.ndarray, power: np.ndarray) -> pd.DataFrame:
    min_gap = DETECTION_MERGE_GAP_SECONDS
    min_dur = DETECTION_MIN_EVENT_SECONDS
    from scipy.ndimage import binary_closing
    is_flare = binary_closing(is_flare, structure=np.ones(min(5, max(1, min_dur // 2))))
    is_flare_int = is_flare.astype(np.int32)
    padded = np.concatenate([[0], is_flare_int, [0]])
    rises = np.where(np.diff(padded) == 1)[0]
    falls = np.where(np.diff(padded) == -1)[0]
    if len(rises) == 0 or len(falls) == 0:
        return pd.DataFrame()
    events = []
    for start, end in zip(rises, falls):
        dur = end - start
        if dur < min_dur:
            continue
        seg = flux[start:end]
        pi = start + int(np.argmax(seg))
        event = {
            "start_time": times[start],
            "peak_time": times[pi],
            "end_time": times[min(end - 1, len(times) - 1)],
            "duration_seconds": float(
                (times[min(end - 1, len(times) - 1)] - times[start]).total_seconds()
            ),
            "peak_flux": float(flux[pi]),
            "peak_power": float(power[pi]),
            "total_fluence": float(np.trapezoid(seg)),
            "peak_excess": float(excess[pi]),
        }
        events.append(event)
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if len(df) > 1:
        return _merge_events(df, gap_seconds=min_gap)
    return df


def _merge_events(df: pd.DataFrame, gap_seconds: int = 30) -> pd.DataFrame:
    df = df.sort_values("start_time").reset_index(drop=True)
    merged = [df.iloc[0].to_dict()]
    for i in range(1, len(df)):
        prev = merged[-1]
        curr = df.iloc[i]
        gap = (curr["start_time"] - prev["end_time"]).total_seconds()
        if gap <= gap_seconds:
            if curr["peak_flux"] > prev["peak_flux"]:
                prev["peak_time"] = curr["peak_time"]
                prev["peak_flux"] = curr["peak_flux"]
                prev["peak_power"] = curr["peak_power"]
                prev["peak_excess"] = curr["peak_excess"]
            prev["end_time"] = curr["end_time"]
            prev["duration_seconds"] = (prev["end_time"] - prev["start_time"]).total_seconds()
            prev["total_fluence"] = prev["total_fluence"] + curr["total_fluence"]
        else:
            merged.append(curr.to_dict())
    return pd.DataFrame(merged)


class WaveletFlareDetector:
    def __init__(self, wavelet: str = "morl", max_scale: int = 64,
                 threshold_sigma: float = DETECTION_THRESHOLD_SIGMA):
        self.wavelet = wavelet
        self.max_scale = max_scale
        self.threshold_sigma = threshold_sigma

    def detect(self, flux_series: pd.Series) -> pd.DataFrame:
        flux = flux_series.values.astype(float)
        times = flux_series.index
        cleaned = flux.copy()
        events = cwt_detect(cleaned, times, self.wavelet,
                            self.max_scale, self.threshold_sigma)
        if len(events) == 0:
            events = _detect_by_excess(cleaned, times, self.threshold_sigma)
        return events
