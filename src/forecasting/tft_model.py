import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss, CrossEntropy, MAE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix,
                             average_precision_score, brier_score_loss)
from src.config import FORECAST_HORIZONS, FORECAST_LOOKBACK_MINUTES, MODELS_DIR


class TFTForecaster:
    """TFT for regression (flux prediction)."""
    def __init__(self, horizon_minutes: int = 30,
                 lookback_minutes: int = FORECAST_LOOKBACK_MINUTES):
        self.horizon_minutes = horizon_minutes
        self.lookback_minutes = lookback_minutes
        self.model = None
        self.training = None
        self.validation = None

    def _prepare_data(self, df: pd.DataFrame,
                      time_idx_col: str = "time_idx",
                      group_col: str = "group") -> TimeSeriesDataSet:
        df = df.copy()
        df["group"] = "all"
        df["time_idx"] = np.arange(len(df))
        df = df.fillna(0)
        target = "flux"
        known_categorical = []
        known_continuous = ["time_idx"]
        if "ssn" in df.columns:
            known_continuous.append("ssn")
        sharp_feats = [c for c in df.columns if c.upper() in
                       ["USFLUX", "TOTUSJH", "TOTUSJZ", "R_VALUE",
                        "TOTPOT", "TOTBSQ", "MEANPOT", "MEANSHR"]]
        known_continuous.extend([c for c in sharp_feats if c in df.columns])
        self._feature_cols = known_continuous
        max_encoder = self.lookback_minutes * 60
        max_pred = self.horizon_minutes * 60
        training = TimeSeriesDataSet(
            df, time_idx="time_idx", target=target, group_ids=[group_col],
            min_encoder_length=max_encoder // 2, max_encoder_length=max_encoder,
            min_prediction_length=1, max_prediction_length=max_pred,
            time_varying_known_reals=known_continuous,
            time_varying_unknown_reals=[target],
            add_relative_time_idx=True, add_target_scales=True,
            add_encoder_length=True,
        )
        return training

    def train(self, df: pd.DataFrame, max_epochs: int = 30,
              batch_size: int = 64, gpu: bool = None) -> dict:
        if gpu is None:
            gpu = torch.cuda.is_available()
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        self.training = self._prepare_data(train_df)
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, val_df, stop_randomization=True
        )
        train_loader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = self.validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        self.model = TemporalFusionTransformer.from_dataset(
            self.training, learning_rate=0.001, hidden_size=64,
            attention_head_size=4, dropout=0.1, hidden_continuous_size=32,
            output_size=7, loss=QuantileLoss(), reduce_on_plateau_patience=4,
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")
        checkpoint = ModelCheckpoint(
            dirpath=str(MODELS_DIR / "tft_checkpoints"),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss", mode="min", save_top_k=1,
        )
        trainer = Trainer(
            max_epochs=max_epochs, accelerator="gpu" if gpu else "cpu",
            devices=1, callbacks=[early_stop, checkpoint],
            enable_progress_bar=False,
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return {"best_val_loss": float(checkpoint.best_model_score or -1)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        loader = self.training.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw_preds, _ = self.model.predict(loader, mode="raw", return_x=True)
        return raw_preds.cpu().numpy()

    def save(self, path=None):
        if self.model is None:
            raise ValueError("No model to save")
        path = path or MODELS_DIR / f"tft_{self.horizon_minutes}min.pt"
        torch.save(self.model.state_dict(), path)
        return path


class TFTBinaryClassifier:
    """TFT for binary classification (flare / no-flare)."""
    def __init__(self, horizon_minutes: int = 60,
                 lookback_minutes: int = FORECAST_LOOKBACK_MINUTES):
        self.horizon_minutes = horizon_minutes
        self.lookback_minutes = lookback_minutes
        self.model = None
        self.training = None
        self.validation = None
        self._feature_cols = []

    def _prepare_data(self, df: pd.DataFrame,
                      time_idx_col: str = "time_idx",
                      group_col: str = "group") -> TimeSeriesDataSet:
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df["group"] = "all"
        df["time_idx"] = np.arange(len(df))
        df = df.fillna(0)
        target = "flare_label"
        known_continuous = ["time_idx", "flux"]
        if "ssn" in df.columns:
            known_continuous.append("ssn")
        sharp_feats = [c for c in df.columns if c.upper() in
                       ["USFLUX", "TOTUSJH", "TOTUSJZ", "R_VALUE",
                        "TOTPOT", "AREA_ACR", "MEANPOT", "MEANGBZ",
                        "SHRGT45", "TOTFZD"]]
        known_continuous.extend([c for c in sharp_feats if c in df.columns])
        known_continuous = list(set(known_continuous) - {target})
        self._feature_cols = known_continuous
        max_encoder = self.lookback_minutes
        max_pred = self.horizon_minutes
        training = TimeSeriesDataSet(
            df, time_idx="time_idx", target=target, group_ids=[group_col],
            min_encoder_length=max(1, max_encoder // 2),
            max_encoder_length=max_encoder,
            min_prediction_length=1, max_prediction_length=max_pred,
            time_varying_known_reals=known_continuous,
            time_varying_unknown_reals=[target],
            add_relative_time_idx=True, add_target_scales=True,
            add_encoder_length=True, categorical_encoders={target: NaNLabelEncoder(add_nan=True)},
        )
        return training

    def train(self, df: pd.DataFrame, max_epochs: int = 30,
              batch_size: int = 64, gpu: bool = None,
              val_df: pd.DataFrame = None) -> dict:
        if gpu is None:
            gpu = torch.cuda.is_available()
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df["group"] = "all"
        df["time_idx"] = np.arange(len(df))
        if val_df is None:
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
        else:
            train_df = df
            val_df = val_df.copy()
            val_df = val_df.loc[:, ~val_df.columns.duplicated()]
            val_df["group"] = "all"
            val_df["time_idx"] = np.arange(len(train_df), len(train_df) + len(val_df))
        self.training = self._prepare_data(train_df)
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, val_df, stop_randomization=True
        )
        train_loader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_loader = self.validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        self.model = TemporalFusionTransformer.from_dataset(
            self.training, learning_rate=0.001, hidden_size=64,
            attention_head_size=4, dropout=0.1, hidden_continuous_size=32,
            output_size=2, loss=CrossEntropy(), reduce_on_plateau_patience=4,
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=5, min_delta=1e-4, mode="min")
        checkpoint = ModelCheckpoint(
            dirpath=str(MODELS_DIR / "tft_classifier"),
            filename="tft_cls-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss", mode="min", save_top_k=1,
        )
        trainer = Trainer(
            max_epochs=max_epochs, accelerator="gpu" if gpu else "cpu",
            devices=1, callbacks=[early_stop, checkpoint],
            enable_progress_bar=False,
        )
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return {"best_val_loss": float(checkpoint.best_model_score or -1)}

    def predict(self, df: pd.DataFrame, training=None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        dataset = training or self.training
        loader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw_preds, _ = self.model.predict(loader, mode="raw", return_x=True)
        return torch.softmax(torch.from_numpy(raw_preds), dim=-1).numpy()

    def evaluate(self, df_test, y_true):
        df = df_test.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        df["group"] = "all"
        df["time_idx"] = np.arange(len(self.validation.data) if hasattr(self, "validation") and hasattr(self.validation, "data") else len(df))
        val_ds = TimeSeriesDataSet.from_dataset(self.training, df, stop_randomization=True)
        y_prob = self.predict(df, training=val_ds)
        y_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        y_pred = (y_prob_pos > 0.5).astype(int)
        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob_pos)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "avg_precision": float(average_precision_score(y_true, y_prob_pos)),
            "brier": float(brier_score_loss(y_true, y_prob_pos)),
        }, y_pred, y_prob_pos
