import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from src.config import FORECAST_HORIZONS


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def train_model(self, train_loader, val_loader, epochs=30, lr=0.001,
                    class_weights=None, device="cpu"):
        self.to(device)
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        best_val = float("inf")
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    loss = criterion(self(Xb), yb)
                    val_loss += loss.item()
            scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
        return {"best_val_loss": best_val}


class XGBoostForecaster:
    def __init__(self, horizon_minutes: int = 30):
        self.horizon = horizon_minutes
        self.model = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = df.copy()
        if "flux" in feats.columns:
            for lag_min in [1, 5, 15, 30, 60]:
                feats[f"flux_lag_{lag_min}"] = feats["flux"].shift(lag_min)
            feats["flux_ma_5"] = feats["flux"].rolling(5, min_periods=1).mean()
            feats["flux_ma_15"] = feats["flux"].rolling(15, min_periods=1).mean()
            feats["flux_ma_30"] = feats["flux"].rolling(30, min_periods=1).mean()
            feats["flux_dt"] = feats["flux"].diff().clip(lower=0)
        return feats.fillna(0)

    def train(self, df: pd.DataFrame, target_col: str = "flare_label",
              class_weight: str = "balanced"):
        feats = self._create_features(df)
        feature_cols = [c for c in feats.columns if c != target_col]
        X = feats[feature_cols].values
        y = feats[target_col].values
        self.feature_cols = feature_cols
        scale_pos = None
        if len(np.unique(y)) == 2:
            ratio = (y == 0).sum() / max((y == 1).sum(), 1)
            scale_pos = ratio
        self.model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
        )
        sample_weight = None
        if class_weight == "balanced" and len(np.unique(y)) > 2:
            classes, counts = np.unique(y, return_counts=True)
            w = {c: len(y) / (len(classes) * max(cn, 1)) for c, cn in zip(classes, counts)}
            sample_weight = np.array([w[yi] for yi in y])
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        feats = self._create_features(df)
        X = feats[self.feature_cols].values
        return self.model.predict_proba(X)

    def forecast(self, df: pd.DataFrame) -> dict:
        probs = self.predict_proba(df.iloc[-1:])
        class_map = {0: "B", 1: "C", 2: "M", 3: "X"}
        result = {"flare_probability": float(1 - probs[0][0])}
        for i, label in class_map.items():
            result[f"p_{label.lower()}"] = float(probs[0][i]) if i < probs.shape[1] else 0.0
        return result
