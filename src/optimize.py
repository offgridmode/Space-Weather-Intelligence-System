import numpy as np
import pandas as pd
import warnings, json, time, sys
from datetime import datetime
from collections import defaultdict
from itertools import product

warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from src.data.solexs_loader import load_solexs_range
from src.data.hel1os_loader import load_hel1os_range
from src.data.goes_loader import load_goes_catalog, get_flare_labels
from src.data.sharp_loader import load_sharp_clean
from src.data.sunspot_loader import load_sunspot
from src.features.physics_features import compute_aligned_physics_features
from src.detection.wavelet_detector import WaveletFlareDetector
from src.config import OUTPUT_DIR

HORIZONS = [15, 30, 60, 120, 180, 240, 300]

# Hyperparameter grid
HP_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
}


def load_detection_features(flux, window_min=60):
    """Count events detected in sliding windows."""
    ws = window_min * 60
    det = WaveletFlareDetector(threshold_sigma=5)
    events = det.detect(flux)
    if len(events) == 0:
        return pd.Series(0, index=flux.index)
    event_counts = pd.Series(0, index=flux.index)
    for _, ev in events.iterrows():
        mask = (flux.index >= ev["start_time"]) & (flux.index <= ev["end_time"])
        event_counts[mask] += 1
    return event_counts.resample("1min").max().reindex(flux.resample("1min").mean().index, method="ffill")


def build_rich_matrix(start_date, end_date):
    print(f"Building rich feature matrix: {start_date} to {end_date}")
    t0 = time.time()

    # SoLEXS
    flux_s = load_solexs_range(start_date, end_date)["flux"]
    flux_1min = flux_s.resample("1min").mean()
    df = pd.DataFrame(index=flux_1min.index)
    df["flux"] = flux_1min.values

    # HEL1OS
    try:
        flux_h = load_hel1os_range(start_date, end_date)["flux"]
        df["hel1os_flux"] = flux_h.resample("1min").mean().reindex(df.index, method="ffill")
        df["hel1os_flux"] = df["hel1os_flux"].fillna(0)
    except Exception:
        print("  HEL1OS: not available")

    # SHARP + sunspot
    try:
        sharp = load_sharp_clean()
        sunspot = load_sunspot()
        aligned = compute_aligned_physics_features(df.index, sharp, sunspot)
        for c in aligned.columns:
            df[c] = aligned[c].values
    except Exception as e:
        print(f"  SHARP/sunspot: {e}")

    # Detection-based features
    try:
        det_flux = flux_s.iloc[-86400:] if len(flux_s) > 86400 else flux_s
        det_counts = load_detection_features(det_flux, window_min=60)
        df["det_events_1h"] = det_counts.reindex(df.index, method="ffill").fillna(0)
        det_counts_3h = load_detection_features(det_flux, window_min=180)
        df["det_events_3h"] = det_counts_3h.reindex(df.index, method="ffill").fillna(0)
    except Exception as e:
        print(f"  Detection features: {e}")

    # GOES labels
    try:
        goes = load_goes_catalog()
        for h in HORIZONS:
            labels = get_flare_labels(goes, df.index, lookback_minutes=h)
            df[f"flare_{h}min"] = labels["flare_within_window"].values
    except Exception as e:
        print(f"  GOES: {e}")

    print(f"  Matrix: {len(df):,} rows, {len(df.columns)} cols  ({time.time()-t0:.1f}s)")
    for h in HORIZONS:
        if f"flare_{h}min" in df.columns:
            pos = df[f"flare_{h}min"].sum()
            print(f"    {h:3d}min: {pos:6.0f}/{len(df):,} = {pos/len(df)*100:.2f}%")
    return df


def _engineered_features(df):
    feats = df.copy()
    f = feats["flux"]
    for lag in [1, 5, 15, 30, 60]:
        feats[f"flux_lag_{lag}"] = f.shift(lag)
    for w in [5, 15, 30, 60]:
        feats[f"flux_ma_{w}"] = f.rolling(w, min_periods=1).mean()
        feats[f"flux_std_{w}"] = f.rolling(w, min_periods=1).std()
    feats["flux_dt"] = f.diff().clip(lower=0)
    feats["flux_dt2"] = feats["flux_dt"].diff().clip(lower=0)
    feats["flux_min"] = f.rolling(60, min_periods=1).min()
    feats["flux_max"] = f.rolling(60, min_periods=1).max()
    feats["flux_range"] = feats["flux_max"] - feats["flux_min"]

    if "hel1os_flux" in feats.columns:
        hf = feats["hel1os_flux"]
        feats["hel1os_lag_5"] = hf.shift(5)
        feats["hel1os_lag_15"] = hf.shift(15)
        feats["hel1os_ma_15"] = hf.rolling(15, min_periods=1).mean()

    # Ratio features
    if "hel1os_flux" in feats.columns:
        feats["hx_ratio"] = feats["hel1os_flux"] / (feats["flux"] + 1e-8)
        feats["hx_excess"] = feats["hel1os_flux"] - feats["flux"].rolling(60, min_periods=1).mean()

    # Rolling flux trend
    feats["flux_slope_30"] = (f.shift(0) - f.shift(30)) / 30
    feats["flux_slope_60"] = (f.shift(0) - f.shift(60)) / 60

    return feats.fillna(0)


def evaluate_xgboost_hp(df, horizon, params, test_ratio=0.2):
    target = f"flare_{horizon}min"
    feats = _engineered_features(df)
    skip_cols = {target, f"class_{horizon}min"}
    skip_cols.update(c for c in feats.columns if c.startswith("flare_") and c != target)
    feature_cols = [c for c in feats.columns if c not in skip_cols]
    X = feats[feature_cols].values
    y = df[target].values.astype(int)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        random_state=42, eval_metric="logloss",
        scale_pos_weight=pos_ratio, **params,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    return {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    }, model


def random_search(df, horizon, n_iter=20):
    target = f"flare_{horizon}min"
    feats = _engineered_features(df)
    skip_cols = {target, f"class_{horizon}min"}
    skip_cols.update(c for c in feats.columns if c.startswith("flare_") and c != target)
    feature_cols = [c for c in feats.columns if c not in skip_cols]
    X = feats[feature_cols].values
    y = df[target].values.astype(int)

    split = int(len(X) * 0.7)
    X_train = X[:split]
    y_train = y[:split]
    pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    best_score = -1
    best_params = None
    best_model = None

    import random
    for i in range(n_iter):
        params = {
            "n_estimators": random.choice(HP_GRID["n_estimators"]),
            "max_depth": random.choice(HP_GRID["max_depth"]),
            "learning_rate": random.choice(HP_GRID["learning_rate"]),
            "subsample": random.choice(HP_GRID["subsample"]),
            "colsample_bytree": random.choice(HP_GRID["colsample_bytree"]),
            "min_child_weight": random.choice(HP_GRID["min_child_weight"]),
            "gamma": random.choice(HP_GRID["gamma"]),
        }
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        model = xgb.XGBClassifier(
            random_state=42, eval_metric="logloss",
            scale_pos_weight=pos_ratio, **params,
        )
        model.fit(X_tr, y_tr)
        yp = model.predict(X_val)
        score = f1_score(y_val, yp, zero_division=0)
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    return best_params, best_score


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "hp_search":
        df = build_rich_matrix("20241101", "20241130")
        for h in [15, 60, 180, 300]:
            print(f"\nHP search for {h}min horizon...")
            t0 = time.time()
            best_params, best_f1 = random_search(df, h, n_iter=30)
            print(f"  Best F1={best_f1:.4f}  params={best_params}  ({time.time()-t0:.1f}s)")

    elif mode == "compare_datasets":
        # Compare 1 month vs 3 months with default params
        for label, start, end in [("1-month", "20241101", "20241130"), ("3-month", "20240901", "20241130")]:
            print(f"\n{'='*50}")
            print(f"DATASET: {label} ({start} to {end})")
            df = build_rich_matrix(start, end)
            for h in HORIZONS:
                m, _ = evaluate_xgboost_hp(df, h, {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0})
                print(f"  {h:3d}min: F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

    elif mode == "best_params":
        # Use found best params and test
        df = build_rich_matrix("20240901", "20241130")
        best_overall = {
            15: {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 1.0, "min_child_weight": 5, "gamma": 0.2},
            30: {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.1, "subsample": 1.0, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0.1},
            60: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "gamma": 0.1},
            120: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0},
            180: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0},
            240: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0},
            300: {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0},
        }
        print(f"\n{'='*70}")
        print("FINAL OPTIMIZED RESULTS (3 months, tuned per horizon)")
        print(f"{'='*70}")
        print(f"{'Horizon':>8} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'ROC-AUC':>8}")
        print("-" * 50)
        for h in HORIZONS:
            params = best_overall.get(h, {})
            m, model = evaluate_xgboost_hp(df, h, params)
            print(f"{h:>8} | {m['f1']:>6.4f} | {m['precision']:>6.4f} | {m['recall']:>6.4f} | {m['roc_auc']:>8.4f}")

        # Feature importance analysis
        print(f"\n{'='*70}")
        print("TOP-10 FEATURES ACROSS HORIZONS")
        print(f"{'='*70}")
        target = "flare_60min"
        feats = _engineered_features(df)
        skip_cols = {target}.union(c for c in feats.columns if c.startswith("flare_") and c != target)
        feature_cols = [c for c in feats.columns if c not in skip_cols]
        _, model = evaluate_xgboost_hp(df, 60, best_overall.get(60, {}))
        imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
        for name, val in imp[:15]:
            print(f"  {name:<25s} {val:.4f}")

    else:
        # Default: comprehensive benchmark
        df = build_rich_matrix("20240901", "20241130")
        default_params = {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "gamma": 0}
        results = {}
        for h in HORIZONS:
            m, _ = evaluate_xgboost_hp(df, h, default_params)
            results[str(h)] = m
            print(f"  {h:3d}min: F1={m['f1']:.4f}  ROC-AUC={m['roc_auc']:.4f}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(OUTPUT_DIR / f"optimized_results_{ts}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\nDone!")


if __name__ == "__main__":
    main()
