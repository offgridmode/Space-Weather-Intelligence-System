import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
import joblib
from src.config import CLASSIFICATION_FEATURES, MODELS_DIR


def extract_event_features(event_row: dict, flux_series: pd.Series = None) -> dict:
    features = {}
    features["peak_flux"] = event_row.get("peak_flux", 0)
    features["duration"] = event_row.get("duration_seconds", 0)
    features["total_fluence"] = event_row.get("total_fluence", 0)
    start = event_row.get("start_time")
    peak = event_row.get("peak_time")
    end = event_row.get("end_time")
    if flux_series is not None and start and peak:
        try:
            pre = flux_series.loc[start - pd.Timedelta(seconds=60):start]
            bg = pre.median() if len(pre) > 0 else 0
        except Exception:
            bg = 0
        features["background_flux"] = bg
        features["peak_to_bg_ratio"] = features["peak_flux"] / max(bg, 1e-12)
        features["rise_time"] = (peak - start).total_seconds() if peak and start else 0
        features["decay_time"] = (end - peak).total_seconds() if end and peak else 0
        features["rise_rate_log10"] = np.log10(
            max(features["peak_flux"] - bg, 1e-12) / max(features["rise_time"], 1)
        ) if features["rise_time"] > 0 else 0
    return features


class FlareClassifier:
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.features = CLASSIFICATION_FEATURES
        self.label_encoder = {"B": 0, "C": 1, "M": 2, "X": 3}

    def _check_features(self, df: pd.DataFrame):
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

    def train(self, df: pd.DataFrame, target_col: str = "flare_class_int"):
        self._check_features(df)
        X = df[self.features].fillna(0).values
        y = df[target_col].values
        X_scaled = self.scaler.fit_transform(X)
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=300, max_depth=15,
                class_weight="balanced", random_state=42, n_jobs=-1
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=300, max_depth=10,
                learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob", num_class=4,
                eval_metric="mlogloss", random_state=42,
            )
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].fillna(0).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].fillna(0).values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, df: pd.DataFrame, target_col: str = "flare_class_int"):
        y_true = df[target_col].values
        y_pred = self.predict(df)
        return {
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    def save(self, path=None):
        path = path or MODELS_DIR / f"classifier_{self.model_type}.joblib"
        joblib.dump({"model": self.model, "scaler": self.scaler,
                      "features": self.features, "model_type": self.model_type}, path)
        return path

    def load(self, path):
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]
        self.model_type = data["model_type"]
        return self
