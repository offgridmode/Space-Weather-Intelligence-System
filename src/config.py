import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# External data (F: drive)
ANTARIKSH_ROOT = Path("F:/Antariksh/PS-15")
SOLEXS_DATA_DIR = ANTARIKSH_ROOT / "data/solexs"
HEL1OS_DATA_DIR = ANTARIKSH_ROOT / "data/hel1os"

# GOES catalog
GOES_CATALOG_DIR = DATA_DIR / "goes_flare_catalog"
GOES_FILES = sorted(GOES_CATALOG_DIR.glob("*.csv"))

# SHARP parameters
SHARP_DIR = DATA_DIR / "sharp_params"
SHARP_FILES = sorted(SHARP_DIR.glob("sharp_2*.csv"))

# Sunspot data
SUNSPOT_FILE = DATA_DIR / "sunspot/SN_d_tot_V2.0.txt"

# Detection parameters
DETECTION_WAVELET = "morl"
DETECTION_SCALES = (1, 128)
DETECTION_THRESHOLD_SIGMA = 5.0
DETECTION_MIN_EVENT_SECONDS = 10
DETECTION_MERGE_GAP_SECONDS = 30

# Classification parameters
CLASSIFICATION_FEATURES = [
    "peak_flux", "duration", "rise_time", "decay_time",
    "total_fluence", "rise_rate_log10", "peak_to_bg_ratio",
]

# Forecasting parameters
FORECAST_HORIZONS = [15, 30, 60]
FORECAST_LOOKBACK_MINUTES = 60

# Model save paths
MODELS_DIR = PROJECT_ROOT / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# MLflow tracking
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlruns"
os.makedirs(MLFLOW_TRACKING_URI, exist_ok=True)
