import numpy as np
import pandas as pd
from src.config import DETECTION_THRESHOLD_SIGMA, DETECTION_MIN_EVENT_SECONDS


class ThresholdFlareDetector:
    def __init__(self, sigma: float = DETECTION_THRESHOLD_SIGMA,
                 min_duration: int = DETECTION_MIN_EVENT_SECONDS,
                 window: int = 3600):
        self.sigma = sigma
        self.min_duration = min_duration
        self.window = window

    def detect(self, flux_series: pd.Series) -> pd.DataFrame:
        flux = flux_series.values.astype(float)
        times = flux_series.index
        baseline = pd.Series(flux).rolling(
            window=self.window, center=True, min_periods=1
        ).median().values
        excess = flux - baseline
        noise_std = np.std(excess[excess < np.percentile(excess, 75)])
        threshold = self.sigma * noise_std
        is_flare = excess > threshold
        padded = np.concatenate([[0], is_flare.astype(int), [0]])
        rises = np.where(np.diff(padded) == 1)[0]
        falls = np.where(np.diff(padded) == -1)[0]
        events = []
        for start, end in zip(rises, falls):
            dur = end - start
            if dur < self.min_duration:
                continue
            seg = flux[start:min(end, len(flux))]
            pi = start + int(np.argmax(seg))
            events.append({
                "start_time": times[start],
                "peak_time": times[pi],
                "end_time": times[min(end - 1, len(times) - 1)],
                "duration_seconds": (times[min(end - 1, len(times) - 1)] - times[start]).total_seconds(),
                "peak_flux": float(flux[pi]),
                "total_fluence": float(np.trapezoid(seg)),
                "peak_excess": float(excess[pi]),
            })
        if not events:
            return pd.DataFrame(columns=["start_time", "peak_time", "end_time",
                                         "duration_seconds", "peak_flux",
                                         "total_fluence", "peak_excess"])
        return pd.DataFrame(events)
