import pandas as pd
import numpy as np


def compute_temporal_features(flux_series: pd.Series,
                              windows_min: list = None) -> pd.DataFrame:
    if windows_min is None:
        windows_min = [1, 5, 15, 30, 60]
    df = pd.DataFrame(index=flux_series.index)
    df["flux"] = flux_series.values
    for w in windows_min:
        window_sec = w * 60
        df[f"ma_{w}min"] = flux_series.rolling(window=window_sec, min_periods=1).mean()
        df[f"std_{w}min"] = flux_series.rolling(window=window_sec, min_periods=1).std()
    df["dF_dt"] = df["flux"].diff().clip(lower=0) / 1.0
    df["d2F_dt2"] = df["dF_dt"].diff().clip(lower=0) / 1.0
    baseline = flux_series.rolling(window=3600, min_periods=1).median()
    df["excess_flux"] = df["flux"] - baseline
    df["excess_ratio"] = df["excess_flux"] / baseline.replace(0, np.nan)
    df["rolling_fluence"] = flux_series.rolling(window=1800, min_periods=1).sum()
    df["flux_gradient_5min"] = (df["ma_5min"] - df["ma_5min"].shift(300)).clip(lower=0)
    df["flux_acceleration"] = df["flux_gradient_5min"].diff().clip(lower=0)
    return df


def compute_lag_features(flux_series: pd.Series,
                         lags_min: list = None) -> pd.DataFrame:
    if lags_min is None:
        lags_min = [1, 5, 10, 15, 30]
    df = pd.DataFrame(index=flux_series.index)
    for lag in lags_min:
        lag_sec = lag * 60
        df[f"lag_{lag}min"] = flux_series.shift(lag_sec)
        df[f"delta_{lag}min"] = flux_series - flux_series.shift(lag_sec)
    return df
