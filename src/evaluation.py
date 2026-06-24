import numpy as np
import pandas as pd
import warnings, json, time, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

import torch
import xgboost as xgb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score, brier_score_loss
)

from src.data.solexs_loader import load_solexs_range
from src.data.goes_loader import load_goes_catalog, get_flare_labels
from src.data.sharp_loader import load_sharp_clean
from src.data.sunspot_loader import load_sunspot
from src.features.physics_features import compute_aligned_physics_features
from src.forecasting.baselines import XGBoostForecaster, LSTMForecaster
from src.config import OUTPUT_DIR

FORECAST_HORIZONS = [15, 30, 60, 120, 180, 240, 300]


def build_feature_matrix(start_date="20241101", end_date="20241130"):
    print(f"Building feature matrix: {start_date} to {end_date}")
    flux = load_solexs_range(start_date, end_date)["flux"]
    flux_1min = flux.resample("1min").mean()
    df = pd.DataFrame(index=flux_1min.index)
    df["flux"] = flux_1min.values

    try:
        sharp = load_sharp_clean()
        sunspot = load_sunspot()
        aligned = compute_aligned_physics_features(df.index, sharp, sunspot)
        for c in aligned.columns:
            df[c] = aligned[c].values
    except Exception as e:
        print(f"  SHARP/sunspot: {e}")

    try:
        goes = load_goes_catalog()
    except Exception as e:
        print(f"  GOES: {e}")
        return df

    for h in FORECAST_HORIZONS:
        labels = get_flare_labels(goes, df.index, lookback_minutes=h)
        df[f"flare_{h}min"] = labels["flare_within_window"].values

    print(f"  Matrix: {len(df):,} rows, {len(df.columns)} cols")
    print(f"  Range:  {df.index[0]} to {df.index[-1]}")
    for h in FORECAST_HORIZONS:
        pos = df[f"flare_{h}min"].sum()
        print(f"    {h:3d}min flare ratio: {pos:6.0f}/{len(df):,} = {pos/max(len(df),1)*100:.2f}%")
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
    return feats.fillna(0)


def evaluate_xgboost(df, horizon, test_ratio=0.2):
    target = f"flare_{horizon}min"
    feats = _engineered_features(df)
    feature_cols = [c for c in feats.columns if c not in (target, f"class_{horizon}min") and not c.startswith("flare_")]
    X = feats[feature_cols].values
    y = df[target].values.astype(int)

    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", scale_pos_weight=pos_ratio,
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return _compute_metrics(y_test, y_pred, y_prob), model, feature_cols


def _create_lstm_sequences(X, y, seq_len=60):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def evaluate_lstm(df, horizon, test_ratio=0.2, seq_len=60):
    target = f"flare_{horizon}min"
    feats = _engineered_features(df)
    feature_cols = [c for c in feats.columns if c not in (target, f"class_{horizon}min") and not c.startswith("flare_")]
    X = feats[feature_cols].values.astype(np.float32)
    y = df[target].values.astype(int)

    split = int(len(X) * (1 - test_ratio))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    X_seq_tr, y_seq_tr = _create_lstm_sequences(X_train, y_train, seq_len)
    X_seq_te, y_seq_te = _create_lstm_sequences(X_test, y_test, seq_len)

    if len(np.unique(y_seq_tr)) < 2 or len(np.unique(y_seq_te)) < 2:
        return {"error": "Only one class in split"}, None

    pos_weight = (y_seq_tr == 0).sum() / max((y_seq_tr == 1).sum(), 1)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_seq_tr), torch.from_numpy(y_seq_tr)),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_seq_te), torch.from_numpy(y_seq_te)),
        batch_size=256, shuffle=False
    )

    model = LSTMForecaster(input_size=X_seq_tr.shape[2], num_classes=2, hidden_size=64, num_layers=2)
    model.train_model(train_loader, val_loader, epochs=10, lr=0.001,
                      class_weights=[1.0, pos_weight], device="cpu")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_seq_te))
        y_prob = torch.softmax(logits, dim=1)[:, 1].numpy()
        y_pred = torch.argmax(logits, dim=1).numpy()
    return _compute_metrics(y_seq_te, y_pred, y_prob), model


def _compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = 0.5
    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["avg_precision"] = 0.0
    try:
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics["brier"] = 1.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    metrics["support"] = int(len(y_true))
    metrics["flare_ratio"] = float(y_true.mean())
    return metrics


def format_report(results):
    lines = []
    lines.append("=" * 130)
    lines.append(f"FLARE FORECASTING BENCHMARK REPORT")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"Data: November 2024, 1-min SoLEXS + SHARP + Sunspot + GOES labels")
    lines.append("=" * 130)
    header = (f"{'Horizon':>8} | {'Model':<10} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | "
              f"{'ROC-AUC':>8} | {'AvgPrec':>8} | {'Brier':>6} | {'TP':>4} | {'FP':>4} | "
              f"{'FN':>4} | {'TN':>5} | {'Flare%':>6} | {'Time':>7}")
    lines.append(header)
    lines.append("-" * 130)

    for h in sorted(results.keys(), key=int):
        for model_type in ["XGBoost", "LSTM"]:
            r = results[h].get(model_type, {})
            if not r or "error" in r:
                lines.append(f"{h:>8} | {model_type:<10} | ERROR: {r.get('error','')[:70]}")
                continue
            t = r.get("_time", 0)
            lines.append(
                f"{h:>8} | {model_type:<10} | {r['f1']:>6.4f} | {r['precision']:>6.4f} | "
                f"{r['recall']:>6.4f} | {r['roc_auc']:>8.4f} | {r['avg_precision']:>8.4f} | "
                f"{r['brier']:>6.4f} | {r['true_positives']:>4d} | {r['false_positives']:>4d} | "
                f"{r['false_negatives']:>4d} | {r['true_negatives']:>5d} | "
                f"{r['flare_ratio']*100:>5.1f}% | {t:>6.1f}s"
            )

    lines.append("=" * 130)
    lines.append("\nBest model per horizon:")
    models_seen = set()
    for h in sorted(results.keys(), key=int):
        best_model = max(results[h], key=lambda m: results[h][m].get("f1", -1))
        best_f1 = results[h][best_model].get("f1", 0)
        best_auc = results[h][best_model].get("roc_auc", 0)
        lines.append(f"  {h:>3}min: {best_model:<10} F1={best_f1:.4f}  ROC-AUC={best_auc:.4f}")
        models_seen.add(best_model)

    lines.append("\n" + "=" * 130)
    return "\n".join(lines)


def main():
    include_lstm = "--lstm" in sys.argv
    df = build_feature_matrix("20241101", "20241130")
    all_results = defaultdict(dict)

    for h in FORECAST_HORIZONS:
        print(f"\n{'='*50}")
        print(f"HORIZON: {h} min")
        print(f"{'='*50}")

        t0 = time.time()
        metrics_xgb, _, _ = evaluate_xgboost(df, h)
        t1 = time.time()
        metrics_xgb["_time"] = t1 - t0
        all_results[str(h)]["XGBoost"] = metrics_xgb
        print(f"  XGBoost: F1={metrics_xgb['f1']:.4f}  ROC-AUC={metrics_xgb['roc_auc']:.4f}  ({t1-t0:.1f}s)")

        if include_lstm:
            t0 = time.time()
            metrics_lstm, _ = evaluate_lstm(df, h)
            t1 = time.time()
            if "error" in metrics_lstm:
                all_results[str(h)]["LSTM"] = metrics_lstm
                print(f"  LSTM:    {metrics_lstm['error']}")
            else:
                metrics_lstm["_time"] = t1 - t0
                all_results[str(h)]["LSTM"] = metrics_lstm
                print(f"  LSTM:    F1={metrics_lstm['f1']:.4f}  ROC-AUC={metrics_lstm['roc_auc']:.4f}  ({t1-t0:.1f}s)")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(OUTPUT_DIR / f"benchmark_partial_{ts}.json", "w") as f:
            json.dump(dict(all_results), f, indent=2, default=str)

    report = format_report(all_results)
    print("\n\n" + report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTPUT_DIR / f"forecast_benchmark_{ts}.txt", "w") as f:
        f.write(report)
    with open(OUTPUT_DIR / f"forecast_benchmark_{ts}.json", "w") as f:
        json.dump(dict(all_results), f, indent=2, default=str)
    print(f"\nSaved to output/forecast_benchmark_{ts}.*")
    return all_results


if __name__ == "__main__":
    main()
