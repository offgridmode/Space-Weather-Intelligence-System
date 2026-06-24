import pandas as pd
import numpy as np
from datetime import timedelta


def match_events(solexs_df: pd.DataFrame,
                 hel1os_df: pd.DataFrame,
                 max_time_diff_seconds: int = 300,
                 min_overlap_ratio: float = 0.5) -> pd.DataFrame:
    solexs = solexs_df.copy().reset_index(drop=True)
    hel1os = hel1os_df.copy().reset_index(drop=True)
    solexs["source"] = "SoLEXS"
    hel1os["source"] = "HEL1OS"
    solexs["matched"] = False
    hel1os["matched"] = False
    matches = []
    unmatched_solexs = []
    for i, se in solexs.iterrows():
        best_match = None
        best_overlap = 0
        for j, he in hel1os.iterrows():
            if he["matched"]:
                continue
            overlap_start = max(se["start_time"], he["start_time"])
            overlap_end = min(se["end_time"], he["end_time"])
            if overlap_end <= overlap_start:
                peak_diff = abs((se["peak_time"] - he["peak_time"]).total_seconds())
                if peak_diff > max_time_diff_seconds:
                    continue
                overlap_ratio = 0
            else:
                overlap_seconds = (overlap_end - overlap_start).total_seconds()
                se_dur = max((se["end_time"] - se["start_time"]).total_seconds(), 1)
                overlap_ratio = overlap_seconds / se_dur
            peak_diff = abs((se["peak_time"] - he["peak_time"]).total_seconds())
            if peak_diff > max_time_diff_seconds and overlap_ratio < min_overlap_ratio:
                continue
            combined_score = overlap_ratio + (1 - peak_diff / max_time_diff_seconds)
            if combined_score > best_overlap:
                best_overlap = combined_score
                best_match = j
        if best_match is not None:
            he = hel1os.loc[best_match]
            matches.append({
                "master_id": f"FL{i:06d}",
                "start_time": min(se["start_time"], he["start_time"]),
                "peak_time": se["peak_time"] if se["peak_flux"] >= he["peak_flux"] else he["peak_time"],
                "end_time": max(se["end_time"], he["end_time"]),
                "duration_seconds": (max(se["end_time"], he["end_time"]) - min(se["start_time"], he["start_time"])).total_seconds(),
                "peak_flux_solexs": se["peak_flux"],
                "peak_flux_hel1os": he["peak_flux"],
                "confidence": "high",
                "source": "both",
                "solexs_start": se["start_time"],
                "hel1os_start": he["start_time"],
            })
            solexs.at[i, "matched"] = True
            hel1os.at[best_match, "matched"] = True
        else:
            unmatched_solexs.append(i)
    for i in unmatched_solexs:
        se = solexs.loc[i]
        matches.append({
            "master_id": f"FL{i:06d}",
            "start_time": se["start_time"],
            "peak_time": se["peak_time"],
            "end_time": se["end_time"],
            "duration_seconds": se["duration_seconds"],
            "peak_flux_solexs": se["peak_flux"],
            "peak_flux_hel1os": np.nan,
            "confidence": "medium",
            "source": "solexs_only",
            "solexs_start": se["start_time"],
            "hel1os_start": pd.NaT,
        })
    for j, he in hel1os.iterrows():
        if not he["matched"]:
            matches.append({
                "master_id": f"FL{j + len(solexs):06d}",
                "start_time": he["start_time"],
                "peak_time": he["peak_time"],
                "end_time": he["end_time"],
                "duration_seconds": he["duration_seconds"],
                "peak_flux_solexs": np.nan,
                "peak_flux_hel1os": he["peak_flux"],
                "confidence": "medium",
                "source": "hel1os_only",
                "solexs_start": pd.NaT,
                "hel1os_start": he["start_time"],
            })
    if not matches:
        return pd.DataFrame()
    catalog = pd.DataFrame(matches)
    catalog = catalog.sort_values("start_time").reset_index(drop=True)
    catalog["master_id"] = [f"FL{i:06d}" for i in range(len(catalog))]
    return catalog
