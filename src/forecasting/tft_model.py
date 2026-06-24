import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.config import FORECAST_HORIZONS, FORECAST_LOOKBACK_MINUTES, MODELS_DIR


class TFTForecaster:
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
        static_categorical = []
        static_reals = []
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
            df,
            time_idx="time_idx",
            target=target,
            group_ids=[group_col],
            min_encoder_length=max_encoder // 2,
            max_encoder_length=max_encoder,
            min_prediction_length=1,
            max_prediction_length=max_pred,
            static_categoricals=static_categorical,
            static_reals=static_reals,
            time_varying_known_categoricals=known_categorical,
            time_varying_known_reals=known_continuous,
            time_varying_unknown_reals=[target],
            add_relative_time_idx=True,
            add_target_scales=True,
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
        train_loader = self.training.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0
        )
        val_loader = self.validation.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0
        )
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.001,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,
            loss=QuantileLoss(),
            reduce_on_plateau_patience=4,
        )
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, min_delta=1e-4, mode="min"
        )
        checkpoint = ModelCheckpoint(
            dirpath=str(MODELS_DIR / "tft_checkpoints"),
            filename="tft-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss", mode="min", save_top_k=1,
        )
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if gpu else "cpu",
            devices=1 if gpu else None,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=False,
        )
        trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        return {"best_val_loss": float(checkpoint.best_model_score or -1)}

    def predict(self, df: pd.DataFrame, n_steps: int = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        if n_steps is None:
            n_steps = self.horizon_minutes * 60
        loader = self.training.to_dataloader(train=False, batch_size=64, num_workers=0)
        raw_preds, x_info = self.model.predict(loader, mode="raw", return_x=True)
        return raw_preds.cpu().numpy()

    def predict_proba(self, df: pd.DataFrame) -> dict:
        preds = self.predict(df)
        mean_pred = float(np.mean(preds)) if preds.size > 0 else 0.0
        p_flare = min(1.0, max(0.0, mean_pred / 1e-5))
        b_prob = p_flare * 0.4
        c_prob = p_flare * 0.35
        m_prob = p_flare * 0.2
        x_prob = p_flare * 0.05
        total = b_prob + c_prob + m_prob + x_prob
        if total > 0:
            b_prob /= total
            c_prob /= total
            m_prob /= total
            x_prob /= total
        return {
            "flare_probability": p_flare,
            "p_b": b_prob * p_flare,
            "p_c": c_prob * p_flare,
            "p_m": m_prob * p_flare,
            "p_x": x_prob * p_flare,
        }

    def save(self, path=None):
        if self.model is None:
            raise ValueError("No model to save")
        path = path or MODELS_DIR / f"tft_{self.horizon_minutes}min.pt"
        torch.save(self.model.state_dict(), path)
        return path

    def load(self, path, dataset=None):
        if dataset is not None:
            self.training = dataset
            self.model = TemporalFusionTransformer.from_dataset(
                self.training, learning_rate=0.001, hidden_size=64,
                attention_head_size=4, dropout=0.1, hidden_continuous_size=32,
                output_size=7, loss=QuantileLoss(),
            )
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
        return self
