import pandas as pd
from pathlib import Path
from src.config import GOES_FILES


def _parse_flux(val):
    import re
    if pd.isna(val):
        return None
    val = str(val).strip()
    m = re.match(r"([\d.]+)([eE][+-]?\d+)?", val)
    if m:
        return float(m.group(0))
    return None


def _class_to_int(flare_class: str) -> int:
    if pd.isna(flare_class):
        return -1
    c = str(flare_class).strip().upper()
    if not c:
        return -1
    class_map = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}
    prefix = c[0]
    return class_map.get(prefix, -1)


def load_goes_catalog() -> pd.DataFrame:
    rows = []
    for f in GOES_FILES:
        df = pd.read_csv(f, skipinitialspace=True)
        cols = [c.lower() for c in df.columns]
        rename = {}
        for orig, low in zip(df.columns, cols):
            if "start" in low and "time" in low:
                rename[orig] = "start_time"
            elif low == "time":
                rename[orig] = "peak_time"
            elif "end" in low and "time" in low:
                rename[orig] = "end_time"
            elif "class" in low:
                rename[orig] = "flare_class"
            elif low in ("xrsb_irrad",) or "flux" in low:
                rename[orig] = "peak_flux"
        df = df.rename(columns=rename)
        needed = ["start_time", "peak_time", "end_time", "flare_class", "peak_flux"]
        if not all(c in df.columns for c in needed[:2]):
            continue
        rows.append(df[needed])
    if not rows:
        raise FileNotFoundError("No GOES catalog files with recognized columns")
    catalog = pd.concat(rows, ignore_index=True)
    catalog["start_time"] = pd.to_datetime(catalog["start_time"], errors="coerce")
    catalog["peak_time"] = pd.to_datetime(catalog["peak_time"], errors="coerce")
    catalog["end_time"] = pd.to_datetime(catalog["end_time"], errors="coerce")
    catalog["peak_flux"] = catalog["peak_flux"].apply(_parse_flux)
    catalog = catalog.dropna(subset=["start_time", "flare_class"])
    catalog["class_int"] = catalog["flare_class"].apply(_class_to_int)
    catalog = catalog.sort_values("start_time").reset_index(drop=True)
    return catalog


def get_flare_labels(catalog: pd.DataFrame,
                     times: pd.DatetimeIndex,
                     lookback_minutes: int = 60) -> pd.DataFrame:
    labels = pd.DataFrame(index=times)
    labels["flare_within_window"] = 0
    labels["max_class"] = -1
    labels["max_class_int"] = -1
    for i, t in enumerate(times):
        window_end = t + pd.Timedelta(minutes=lookback_minutes)
        mask = ((catalog["start_time"] >= t) & (catalog["start_time"] < window_end))
        in_window = catalog[mask]
        if len(in_window) > 0:
            labels.loc[times[i], "flare_within_window"] = 1
            labels.loc[times[i], "max_class_int"] = in_window["class_int"].max()
            class_map = {0: "A", 1: "B", 2: "C", 3: "M", 4: "X"}
            labels.loc[times[i], "max_class"] = class_map.get(
                labels.loc[times[i], "max_class_int"], "N"
            )
    return labels
