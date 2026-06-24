import pandas as pd
import numpy as np
from pathlib import Path
from src.config import SHARP_DIR, SHARP_FILES


SHARP_FEATURES = [
    "USFLUX", "TOTUSJH", "TOTUSJZ", "R_VALUE",
    "AREA_ACR", "TOTPOT", "MEANPOT", "MEANGBZ",
    "SHRGT45", "MEANGAM", "MEANALP", "TOTFZD",
    "EPSY", "EPSZ", "SAVNCPP", "MEANJZD",
]


def load_sharp_all() -> pd.DataFrame:
    frames = []
    for f in SHARP_FILES:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError("No SHARP parameter files found")
    return pd.concat(frames, ignore_index=True)


def load_sharp_clean() -> pd.DataFrame:
    df = load_sharp_all()
    cols = [c.upper() for c in df.columns]
    rename = dict(zip(df.columns, cols))
    df = df.rename(columns=rename)
    time_col = None
    for c in cols:
        if c == "T_REC" or "DATE" in c or "TIME" in c:
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]
    df = df.rename(columns={time_col: "timestamp"})
    ts = df["timestamp"].astype(str).str.replace("_TAI", "", regex=False)
    df["timestamp"] = pd.to_datetime(ts, format="%Y.%m.%d_%H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    available = [c for c in SHARP_FEATURES if c in df.columns]
    keep = ["timestamp"] + available
    df = df[keep].copy()
    for c in available:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def compute_physics_features(sharp_df: pd.DataFrame) -> pd.DataFrame:
    df = sharp_df.copy()
    available = df.columns.tolist()
    if "USFLUX" in available and "TOTUSJH" in available:
        df["magnetic_instability_index"] = df["USFLUX"].abs() * df["TOTUSJH"].abs()
    if "TOTPOT" in available and "AREA_ACR" in available:
        df["free_energy_proxy"] = df["TOTPOT"].fillna(0) - df["AREA_ACR"].fillna(0)
    if "TOTUSJZ" in available and "TOTUSJH" in available:
        denom = df["TOTUSJH"].replace(0, np.nan)
        df["magnetic_twist"] = df["TOTUSJZ"].abs() / denom.abs()
    if "MEANGBZ" in available:
        df["mean_shear"] = df["MEANGBZ"]
    return df
