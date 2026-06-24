import zipfile
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from src.config import HEL1OS_DATA_DIR

warnings.filterwarnings("ignore", category=UserWarning)


def list_available() -> list:
    files = sorted(HEL1OS_DATA_DIR.glob("HLS_*.zip"))
    entries = []
    for f in files:
        parts = f.stem.split("_")
        if len(parts) >= 2:
            entries.append((parts[1][:8], f))
    return entries


def load_hel1os_day(date_str: str,
                    detector: str = "czt",
                    detector_num: str = "1",
                    energy_band: str = "20.00KEV_TO_40.00KEV") -> pd.DataFrame:
    files = sorted(HEL1OS_DATA_DIR.glob(f"HLS_{date_str}*.zip"))
    if not files:
        raise FileNotFoundError(f"No HEL1OS data for {date_str}")
    f = files[0]
    with zipfile.ZipFile(f) as z:
        lc_path = None
        for n in z.namelist():
            if f"lightcurve_{detector}{detector_num}.fits" in n:
                lc_path = n
                break
        if lc_path is None:
            raise FileNotFoundError(f"Light curve not found for {detector}{detector_num}")
        raw = z.read(lc_path)
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
    tmp.write(raw)
    tmp.close()
    try:
        with fits.open(tmp.name) as hdul:
            ext_name = None
            for hdu in hdul:
                if isinstance(hdu, fits.BinTableHDU):
                    if ext_name is None:
                        ext_name = hdu.name
                    if detector.upper() in hdu.name and energy_band.replace("KEV", "KEV") in hdu.name:
                        ext_name = hdu.name
                        break
            if ext_name is None:
                raise ValueError(f"No matching extension in HEL1OS FITS")
            data = hdul[ext_name].data
            df = pd.DataFrame({
                "time_mjd": data["MJD"],
                "time_iso": data["ISOT"],
                "flux": data["CTR"].astype(float),
                "flux_err": data["STAT_ERR"].astype(float),
            })
    finally:
        try:
            os.unlink(tmp.name)
        except PermissionError:
            pass
    df["time_iso"] = pd.to_datetime(df["time_iso"])
    df = df.set_index("time_iso").drop(columns=["time_mjd"], errors="ignore")
    return df


def load_hel1os_range(start_date: str, end_date: str,
                      detector: str = "czt",
                      detector_num: str = "1",
                      energy_band: str = "20.00KEV_TO_40.00KEV") -> pd.DataFrame:
    frames = []
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    for d in dates:
        ds = d.strftime("%Y%m%d")
        try:
            df = load_hel1os_day(ds, detector=detector,
                                 detector_num=detector_num,
                                 energy_band=energy_band)
            frames.append(df)
        except (FileNotFoundError, ValueError):
            continue
    if not frames:
        raise FileNotFoundError(f"No HEL1OS data from {start_date} to {end_date}")
    return pd.concat(frames).sort_index()
