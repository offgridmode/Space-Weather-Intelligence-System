import pandas as pd
from src.config import SUNSPOT_FILE


def load_sunspot() -> pd.DataFrame:
    col_names = ["year", "month", "day", "fdate", "ssn", "ssn_sd",
                 "obs_count", "is_definitive"]
    df = pd.read_csv(
        SUNSPOT_FILE,
        delim_whitespace=True,
        comment="#",
        names=col_names,
        na_values=["-1", "-1.0"],
    )
    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day"]].astype(str).agg("-".join, axis=1),
        errors="coerce",
    )
    df["ssn"] = pd.to_numeric(df["ssn"], errors="coerce")
    df = df.dropna(subset=["timestamp", "ssn"])
    df = df.set_index("timestamp").sort_index()
    df = df[["ssn"]]
    return df
