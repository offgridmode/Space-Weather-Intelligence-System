import numpy as np
import pandas as pd
from src.detection.wavelet_detector import WaveletFlareDetector, compute_baseline
from src.classification.classifier import FlareClassifier, extract_event_features
from src.config import DETECTION_THRESHOLD_SIGMA


class Nowcaster:
    def __init__(self, detector=None, classifier=None):
        self.detector = detector or WaveletFlareDetector(
            threshold_sigma=DETECTION_THRESHOLD_SIGMA * 0.8
        )
        self.classifier = classifier
        self.buffer_seconds = 300
        self.current_state = {
            "flare_active": False,
            "flare_class": None,
            "confidence": 0.0,
            "start_time": None,
            "peak_flux": 0.0,
            "time_since_onset": 0.0,
        }

    def update(self, flux_series: pd.Series) -> dict:
        if len(flux_series) < 10:
            return self.current_state
        recent = flux_series.iloc[-self.buffer_seconds:] if len(flux_series) > self.buffer_seconds else flux_series
        events = self.detector.detect(recent)
        now = flux_series.index[-1]
        active_events = events[events["end_time"] >= now] if len(events) > 0 else pd.DataFrame()
        if len(active_events) > 0:
            event = active_events.iloc[-1]
            self.current_state["flare_active"] = True
            self.current_state["start_time"] = event["start_time"]
            self.current_state["peak_flux"] = float(event["peak_flux"])
            self.current_state["time_since_onset"] = (now - event["start_time"]).total_seconds()
            if self.classifier and len(active_events) > 0:
                features = extract_event_features(event.to_dict(), flux_series)
                feat_df = pd.DataFrame([features])
                try:
                    probs = self.classifier.predict_proba(feat_df)
                    class_idx = np.argmax(probs[0])
                    class_map = {0: "B", 1: "C", 2: "M", 3: "X"}
                    self.current_state["flare_class"] = class_map.get(class_idx, "?")
                    self.current_state["confidence"] = float(np.max(probs[0]))
                except Exception:
                    pass
            self.current_state["peak_flux"] = max(
                self.current_state["peak_flux"], float(event["peak_flux"])
            )
        else:
            if len(events) > 0:
                last_event = events.iloc[-1]
                time_since_end = (now - last_event["end_time"]).total_seconds()
                if time_since_end < 60:
                    return self.current_state
            self.current_state["flare_active"] = False
            self.current_state["time_since_onset"] = 0.0
        return self.current_state

    def get_status(self) -> dict:
        return self.current_state
