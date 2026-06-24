import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from src.config import OUTPUT_DIR, FORECAST_HORIZONS
from src.data.solexs_loader import load_solexs_range
from src.data.hel1os_loader import load_hel1os_range
from src.data.goes_loader import load_goes_catalog, get_flare_labels
from src.data.sharp_loader import load_sharp_clean, compute_physics_features
from src.data.sunspot_loader import load_sunspot
from src.detection.wavelet_detector import WaveletFlareDetector
from src.detection.threshold_detector import ThresholdFlareDetector
from src.classification.classifier import FlareClassifier
from src.catalog.master_catalog import match_events
from src.features.temporal_features import compute_temporal_features
from src.features.physics_features import compute_aligned_physics_features
from src.nowcasting.nowcaster import Nowcaster
from src.forecasting.tft_model import TFTForecaster
from src.forecasting.baselines import XGBoostForecaster


class SpaceWeatherPipeline:
    def __init__(self):
        self.solexs_data = None
        self.hel1os_data = None
        self.goes_catalog = None
        self.sharp_data = None
        self.sunspot_data = None
        self.solexs_events = None
        self.hel1os_events = None
        self.master_catalog = None
        self.detector = WaveletFlareDetector()
        self.classifier = None
        self.nowcaster = Nowcaster()
        self.forecaster = None

    def load_all_data(self, start_date: str, end_date: str):
        print(f"Loading data: {start_date} to {end_date}")
        try:
            self.solexs_data = load_solexs_range(start_date, end_date)
            print(f"  SoLEXS: {len(self.solexs_data)} rows")
        except FileNotFoundError as e:
            print(f"  SoLEXS: {e}")
        try:
            self.hel1os_data = load_hel1os_range(start_date, end_date)
            print(f"  HEL1OS: {len(self.hel1os_data)} rows")
        except FileNotFoundError as e:
            print(f"  HEL1OS: {e}")
        try:
            self.goes_catalog = load_goes_catalog()
            print(f"  GOES catalog: {len(self.goes_catalog)} events")
        except FileNotFoundError:
            print("  GOES catalog: not found")
        try:
            self.sharp_data = load_sharp_clean()
            print(f"  SHARP: {len(self.sharp_data)} rows")
        except FileNotFoundError:
            print("  SHARP: not found")
        try:
            self.sunspot_data = load_sunspot()
            print(f"  Sunspot: {len(self.sunspot_data)} rows")
        except FileNotFoundError:
            print("  Sunspot: not found")

    def run_detection(self):
        print("\nRunning detection...")
        if self.solexs_data is not None:
            print("  Detecting on SoLEXS...")
            self.solexs_events = self.detector.detect(
                self.solexs_data["flux"]
            )
            print(f"    Found {len(self.solexs_events)} events")
        if self.hel1os_data is not None:
            print("  Detecting on HEL1OS...")
            self.hel1os_events = self.detector.detect(
                self.hel1os_data["flux"]
            )
            print(f"    Found {len(self.hel1os_events)} events")
        results = {}
        if self.solexs_events is not None:
            results["solexs"] = len(self.solexs_events)
        if self.hel1os_events is not None:
            results["hel1os"] = len(self.hel1os_events)
        return results

    def run_classification(self):
        print("\nRunning classification...")
        if self.master_catalog is None or len(self.master_catalog) == 0:
            print("  No events to classify")
            return {}
        data = {"flux": self.solexs_data["flux"] if self.solexs_data is not None else None}
        features_list = []
        for _, event in self.master_catalog.iterrows():
            feats = {
                "peak_flux": event.get("peak_flux_solexs", 0),
                "duration": event.get("duration_seconds", 0),
                "rise_time": 0,
                "decay_time": 0,
                "total_fluence": 0,
                "rise_rate_log10": 0,
                "peak_to_bg_ratio": 1,
            }
            if data["flux"] is not None:
                try:
                    start = event["start_time"]
                    peak = event["peak_time"]
                    end = event["end_time"]
                    pre = data["flux"].loc[start - pd.Timedelta(seconds=60):start]
                    bg = pre.median() if len(pre) > 0 else 0
                    feats["peak_to_bg_ratio"] = feats["peak_flux"] / max(bg, 1e-12)
                    feats["rise_time"] = (peak - start).total_seconds() if pd.notna(peak) and pd.notna(start) else 0
                    feats["decay_time"] = (end - peak).total_seconds() if pd.notna(end) and pd.notna(peak) else 0
                except Exception:
                    pass
            features_list.append(feats)
        feat_df = pd.DataFrame(features_list)
        if self.goes_catalog is not None:
            labels = []
            for _, event in self.master_catalog.iterrows():
                mask = (
                    (self.goes_catalog["start_time"] >= event["start_time"] - pd.Timedelta(hours=1))
                    & (self.goes_catalog["start_time"] <= event["end_time"] + pd.Timedelta(hours=1))
                )
                matched = self.goes_catalog[mask]
                if len(matched) > 0:
                    labels.append(matched.iloc[0]["class_int"])
                else:
                    labels.append(-1)
            feat_df["flare_class_int"] = labels
            labeled = feat_df[feat_df["flare_class_int"] >= 1]
            if len(labeled) >= 10:
                self.classifier = FlareClassifier(model_type="random_forest")
                self.classifier.train(labeled)
                metrics = self.classifier.evaluate(labeled)
                print(f"    Macro F1: {metrics['macro_f1']:.3f}")
                print(f"    Weighted F1: {metrics['weighted_f1']:.3f}")
                return metrics
        print("    Insufficient labeled events")
        return {}

    def run_catalog_merge(self):
        print("\nMerging catalogs...")
        s_events = self.solexs_events
        h_events = self.hel1os_events
        if s_events is None or h_events is None:
            df = s_events if s_events is not None else h_events
            if df is not None and len(df) > 0:
                self.master_catalog = df.copy()
                self.master_catalog["master_id"] = [f"FL{i:06d}" for i in range(len(df))]
                print(f"    Single-source catalog: {len(self.master_catalog)} events")
                return len(self.master_catalog)
            print("    No events to merge")
            return 0
        self.master_catalog = match_events(s_events, h_events)
        print(f"    Master catalog: {len(self.master_catalog)} events")
        return len(self.master_catalog)

    def run_nowcasting(self):
        print("\nRunning nowcasting...")
        flux = None
        if self.solexs_data is not None:
            flux = self.solexs_data["flux"]
        elif self.hel1os_data is not None:
            flux = self.hel1os_data["flux"]
        if flux is None:
            print("  No flux data available")
            return {}
        if self.classifier is not None:
            self.nowcaster.classifier = self.classifier
        status = self.nowcaster.update(flux)
        print(f"    Flare active: {status['flare_active']}")
        if status['flare_active']:
            print(f"    Class: {status['flare_class']}, Confidence: {status['confidence']:.3f}")
        return status

    def run_forecasting(self, horizon_minutes: int = 30):
        print(f"\nRunning forecasting (horizon={horizon_minutes}min)...")
        flux = None
        if self.solexs_data is not None:
            flux = self.solexs_data["flux"]
        elif self.hel1os_data is not None:
            flux = self.hel1os_data["flux"]
        if flux is None:
            return {}
        flux_ds = flux.resample("1min").mean()
        df = pd.DataFrame(index=flux_ds.index)
        df["flux"] = flux_ds.values
        sunspot = self.sunspot_data if self.sunspot_data is not None and not self.sunspot_data.empty else pd.DataFrame()
        if self.sharp_data is not None:
            aligned = compute_aligned_physics_features(
                flux_ds.index, self.sharp_data, sunspot
            )
            for c in aligned.columns:
                df[c] = aligned[c].values
        if self.goes_catalog is not None:
            labels = get_flare_labels(
                self.goes_catalog, flux_ds.index, lookback_minutes=horizon_minutes
            )
            df["flare_label"] = labels["flare_within_window"].values
        else:
            df["flare_label"] = 0
        forecaster = XGBoostForecaster(horizon_minutes=horizon_minutes)
        forecaster.train(df)
        forecast = forecaster.forecast(df)
        print(f"    P(flare): {forecast['flare_probability']:.4f}")
        for k in ["p_b", "p_c", "p_m", "p_x"]:
            if k in forecast:
                print(f"    P({k[-1].upper()}): {forecast[k]:.4f}")
        return forecast

    def run_all(self, start_date: str, end_date: str):
        results = {}
        self.load_all_data(start_date, end_date)
        results["detection"] = self.run_detection()
        results["catalog_size"] = self.run_catalog_merge()
        results["classification"] = self.run_classification()
        results["nowcasting"] = self.run_nowcasting()
        results["forecasting"] = {}
        for h in FORECAST_HORIZONS:
            results["forecasting"][f"{h}min"] = self.run_forecasting(h)
        return results


def main():
    import sys
    pipeline = SpaceWeatherPipeline()
    start = sys.argv[1] if len(sys.argv) > 1 else "20241101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20241130"
    results = pipeline.run_all(start, end)
    output_path = OUTPUT_DIR / f"pipeline_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    main()
