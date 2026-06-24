import zipfile
import gzip
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits
from src.config import SOLEXS_DATA_DIR

warnings.filterwarnings("ignore", category=UserWarning)


def list_available_dates() -> list:
    files = sorted(SOLEXS_DATA_DIR.glob("AL1_SLX_L1_*.zip"))
    dates = []
    for f in files:
        parts = f.stem.split("_")
        try:
            idx = parts.index("L1") + 1
            date_str = parts[idx][:8]
            dates.append((date_str, f))
        except (ValueError, IndexError):
            continue
    return dates


def load_solexs_day(date_str: str, detector: str = "SDD2") -> pd.DataFrame:
    files = sorted(SOLEXS_DATA_DIR.glob(f"AL1_SLX_L1_{date_str}*.zip"))
    if not files:
        raise FileNotFoundError(f"No SoLEXS data for {date_str}")
    f = files[0]
    with zipfile.ZipFile(f) as z:
        lc_path = None
        for n in z.namelist():
            if detector in n and n.endswith("_L1.lc.gz"):
                lc_path = n
                break
        if lc_path is None:
            raise FileNotFoundError(f"No light curve for {detector} in {f.name}")
        raw = gzip.decompress(z.read(lc_path))
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
    tmp.write(raw)
    tmp.close()
    try:
        with fits.open(tmp.name) as hdul:
            ext = None
            for hdu in hdul:
                if isinstance(hdu, fits.BinTableHDU):
                    ext = hdu.name
                    break
            if ext is None:
                raise ValueError("No binary table found in SoLEXS FITS")
            data = hdul[ext].data
            df = pd.DataFrame({
                "time": data["TIME"],
                "flux": data["COUNTS"].astype(float),
            })
            df["time_iso"] = pd.to_datetime(df["time"], unit="s")
            df = df[df["time_iso"].dt.year > 2020]
            df = df.set_index("time_iso").drop(columns=["time"])
    finally:
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass
    return df


def load_solexs_range(start_date: str, end_date: str,
                      detector: str = "SDD2") -> pd.DataFrame:
    frames = []
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    for d in dates:
        ds = d.strftime("%Y%m%d")
        try:
            df = load_solexs_day(ds, detector=detector)
            frames.append(df)
        except (FileNotFoundError, ValueError):
            continue
    if not frames:
        raise FileNotFoundError(f"No SoLEXS data from {start_date} to {end_date}")
    return pd.concat(frames).sort_index()
