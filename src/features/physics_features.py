import pandas as pd
import numpy as np


def compute_aligned_physics_features(flux_times: pd.DatetimeIndex,
                                     sharp_df: pd.DataFrame,
                                     sunspot_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=flux_times)
    df = df.join(sharp_df, how="left")
    df = df.join(sunspot_df, how="left")
    df = df.ffill(limit=6)
    return df


def compute_energy_accumulation(flux_series: pd.Series,
                                window_min: int = 30) -> pd.Series:
    window_sec = window_min * 60
    return flux_series.rolling(window=window_sec, min_periods=1).sum()


def compute_magnetic_instability(usflux: pd.Series,
                                 totusjh: pd.Series) -> pd.Series:
    return usflux.abs() * totusjh.abs()


def compute_precursor_heating(flux_series: pd.Series,
                              window_pre: int = 300,
                              window_during: int = 60) -> pd.Series:
    pre_baseline = flux_series.rolling(window=window_pre, min_periods=1).median()
    current = flux_series.rolling(window=window_during, min_periods=1).mean()
    return (current - pre_baseline) / pre_baseline.replace(0, np.nan) * 100
