# Space Weather Intelligence System

## ISRO Aditya-L1 Solar Flare Nowcasting & Forecasting System

A research-grade AI system for detecting, classifying, nowcasting, and forecasting solar flares using data from India's Aditya-L1 mission (SoLEXS & HEL1OS payloads), augmented with NOAA GOES catalog, SHARP magnetic parameters, and sunspot data.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Problem Statement (from Official Slides)](#2-problem-statement-from-official-slides)
3. [Project Goals](#3-project-goals)
4. [Project Folder Structure](#4-project-folder-structure)
5. [Data Sources](#5-data-sources)
6. [System Architecture (Big Picture)](#6-system-architecture-big-picture)
7. [Stage 1: Data Engineering](#7-stage-1-data-engineering)
8. [Stage 2: Detection Pipeline](#8-stage-2-detection-pipeline)
9. [Stage 3: Classification Pipeline](#9-stage-3-classification-pipeline)
10. [Stage 4: Master Flare Catalog](#10-stage-4-master-flare-catalog)
11. [Stage 5: Physics-Informed Feature Engineering](#11-stage-5-physics-informed-feature-engineering)
12. [Stage 6: Nowcasting System](#12-stage-6-nowcasting-system)
13. [Stage 7: Forecasting System](#13-stage-7-forecasting-system)
14. [Model Benchmarking Strategy](#14-model-benchmarking-strategy)
15. [Deep Learning Experiments](#15-deep-learning-experiments)
16. [Advanced Temporal Models](#16-advanced-temporal-models)
17. [Transformer-Based Models](#17-transformer-based-models)
18. [Physics-Informed Learning (PINN)](#18-physics-informed-learning-pinn)
19. [Advanced Techniques from Research Papers](#19-advanced-techniques-from-research-papers)
20. [Final Recommended Architecture](#20-final-recommended-architecture)
21. [Evaluation Strategy](#21-evaluation-strategy)
22. [Deployment & Dashboard](#22-deployment--dashboard)
23. [Ablation Study Plan](#23-ablation-study-plan)
24. [Research-Grade Experimentation Plan](#24-research-grade-experimentation-plan)
25. [Implementation Roadmap](#25-implementation-roadmap)

---

## 1. What Is This Project?

Solar flares are sudden, intense bursts of radiation from the Sun. They can disrupt satellites, power grids, and communication systems on Earth. This project builds an **AI-powered system** that:

- **Detects** solar flares as they happen (using SoLEXS and HEL1OS data from Aditya-L1)
- **Classifies** them into B, C, M, X categories (by strength)
- **Nowcasts** (detects in real-time) ongoing flare activity
- **Forecasts** the probability of a flare occurring in the next 15/30/60 minutes

Think of it as a "weather radar" for solar storms — but powered by machine learning and Indian space data.

---

## 2. Problem Statement (from Official Slides)

The problem statement is derived from the official ISRO Aditya-L1 problem statement slides and consists of four independent but connected modules. The architecture must NOT treat this as only a time-series forecasting problem — it is fundamentally **Detection + Classification + Nowcasting + Forecasting** with separate objectives and evaluation criteria.

### Slide 1: Solar Flare Detection

The slide shows a solar X-ray flux time series with key labeled points:
- **Green = Start** of flare
- **Black = Peak** of flare
- **Red = Stop** of flare

**Required outputs** for every detected flare:
- Start Time, Peak Time, End Time, Duration, Peak Flux

**Key requirements:**
- Detection must work on SoLEXS **independently** and HEL1OS **independently**
- Milestone 1: Build detection algorithms for SoLEXS and HEL1OS separately
- Milestone 2: Generate a unified Master Flare Catalog
- Must detect small, medium, and large flares while avoiding false alarms from noise spikes

### Slide 2: Solar Flare Prediction

The slide separates observed data from future data with a vertical dashed line at "Current Time":
- **Left**: Observed Flux History (available to model)
- **Right**: Future/Unseen Region (must be forecast)
- **Highlighted** red region: **Precursor Heating Signature** before major flare onset

**Forecasting objectives - Milestone 1:** Predict probability of a flare occurring in the next N minutes (15, 30, 60, or custom N-minute horizon)

**Forecasting objectives - Milestone 2:** Predict multi-class flare probabilities:
- P(B), P(C), P(M), P(X), and Overall Flare Probability

### Slide 3: Expected Outcomes and Roadmap

**Three expected outcomes:**
1. **Combined SoLEXS and HEL1OS nowcasting catalog** — unified flare event catalog from both instruments
2. **Forecasting system with quantifiable lead time** — measure how early a flare can be predicted before occurrence
3. **Visual Dashboard** — current activity, forecasts, alerts, historical events, confidence scores

**Roadmap from slide:**
1. Download data from ISSDC portal
2. **Characterize SXR/HXR temporal structures** — analyze temporal evolution, energy buildup, flux trends, precursor patterns (not just generic ML)
3. Build unified master flare catalog
4. Train time-series forecasting models

### Slide 4: Resources and Evaluation

**Allowed datasets:**
- Primary: Aditya-L1 SoLEXS Level-1, Aditya-L1 HEL1OS Level-1
- Supplementary (open-source): GOES catalog, SHARP magnetic parameters, Sunspot numbers

**Evaluation requirements:**
- Detection accuracy across low and high class flares
- High True Positive Rate (catch real flares)
- Low False Positive Rate (avoid noise triggers)
- Low Lead Time (useful advance warning)

**Metrics required:**
- **Detection**: Precision, Recall, F1, TPR, FPR
- **Classification**: Macro F1, Weighted F1
- **Forecasting**: ROC-AUC, PR-AUC, Brier Score, Lead Time

### Critical Design Implication

The system is NOT a single time-series forecasting model. It is four modules:

| Module | Input | Output |
|---|---|---|
| **Flare Detection** | Flux time series | Start, Peak, End |
| **Flare Classification** | Detected event features | B/C/M/X class |
| **Nowcasting** | Current observations | Current flare state + confidence |
| **Forecasting** | Historical + physics features | P(flare) in next N min |

All four modules must be designed with their unique objectives and evaluation criteria. All data sources (SoLEXS, HEL1OS, GOES, SHARP, sunspot) play equally important roles in temporal evolution analysis.

---

## 3. Project Goals

### Primary Goals

1. Build independent **detection algorithms** for SoLEXS and HEL1OS data
2. Generate a **unified master flare catalog** by merging detections from both sensors
3. Build a **nowcasting system** that detects ongoing flares in real-time with low false alarms
4. Build a **forecasting system** that predicts flare probability (B/C/M/X) with quantifiable lead time
5. Create a **visual dashboard** showing current status, alerts, and forecasts

### Key Requirements

| Requirement | Target |
|---|---|
| Low lead time forecasting | Predict before flare onset |
| High True Positive Rate (TPR) | Catch real flares |
| Low False Positive Rate (FPR) | Don't trigger on noise |
| Multi-class probability | P(B), P(C), P(M), P(X) |
| Noise immunity | Distinguish flares from spikes |

---

## 4. Project Folder Structure

```
G:\Space-Weather-Intelligence-System\
│
├── README.md                          # Project documentation (this file)
│
├── .gitignore                         # Git ignore rules for ML artifacts & large files
│
├── data/                              # All datasets - raw and processed
│   │
│   ├── goes_flare_catalog/            # NOAA GOES X-ray flare catalog (CSV)
│   │   ├── sci_xrsf-l2-flrpt_geo_y2024_v1-0-0.csv  # 2024 flare events
│   │   ├── sci_xrsf-l2-flrpt_geo_y2025_v1-0-0.csv  # 2025 flare events
│   │   └── sci_xrsf-l2-flrpt_geo_y2026_v1-0-0.csv  # 2026 flare events
│   │   Purpose: Provides ground truth labels (start/peak/end times, class)
│   │   for training detection, classification, and forecasting models.
│   │
│   ├── sharp_params/                  # SHARP magnetic parameters (CSV)
│   │   ├── sharp_all.csv              # Combined SHARP data (all periods)
│   │   ├── sharp_14day_test.csv       # 14-day sample for testing
│   │   └── sharp_YYYYMMDD_YYYYMMDD.csv # Biweekly SHARP parameter files
│   │   Purpose: Provides physics-based magnetic field features
│   │   (USFLUX, TOTUSJH, R_VALUE, etc.) for flare precursor detection.
│   │
│   └── sunspot/                       # Daily sunspot numbers
│       └── SN_d_tot_V2.0.txt          # International sunspot number (daily)
│       Purpose: Provides solar cycle context (global activity level).
│
├── notebooks/                         # Jupyter notebooks (future)
│   (to be created)                    # For EDA, prototyping, and experiments
│
├── src/                               # Source code (future)
│   (to be created)                    # Organized by module:
│       ├── detection/                 #   Wavelet + ML detection algorithms
│       ├── classification/            #   B/C/M/X classifiers
│       ├── nowcasting/                #   Real-time nowcasting engine
│       ├── forecasting/              #   TFT and other forecast models
│       ├── features/                  #   Feature engineering pipeline
│       ├── data/                      #   Loaders, preprocessors, sync
│       └── dashboard/                 #   Streamlit dashboard
│
├── models/                            # Trained model artifacts (future)
│   (to be created, gitignored)
│
├── mlruns/                            # MLflow experiment logs (future)
│   (to be created, gitignored)
│
├── docs/                              # Additional documentation (future)
│   (to be created)
│
└── tests/                             # Unit and integration tests (future)
    (to be created)
```

---

## 5. Data Sources

### Primary (from Aditya-L1)

| Payload | Data Type | What It Measures |
|---|---|---|
| **SoLEXS** | Soft X-ray flux time series | Low-energy X-ray emissions from the Sun |
| **HEL1OS** | Hard X-ray flux time series | High-energy X-ray emissions from the Sun |

### Supplementary (open-source)

| Dataset | What It Provides |
|---|---|
| **NOAA GOES Flare Catalog** | Historical flare start/peak/end times, class (B/C/M/X), peak flux |
| **SHARP Magnetic Parameters** | 7 magnetic field properties of active regions (USFLUX, TOTUSJH, TOTUSJZ, R_VALUE, TOTPOT, TOTBSQ, etc.) |
| **Sunspot Numbers** | Daily count of sunspots (proxy for solar activity level) |

### Why Combine Them?

- **SoLEXS + HEL1OS** give us the primary flare signal from two energy bands
- **GOES catalog** provides ground truth labels for training
- **SHARP parameters** encode the magnetic complexity of active regions (physical cause of flares)
- **Sunspot data** gives the global solar activity context

---

## 6. System Architecture (Big Picture)

```
                      ┌──────────────────────────────────────┐
                      │         DATA INGESTION LAYER          │
                      │  SoLEXS / HEL1OS / GOES / SHARP / Sun │
                      └──────────────┬───────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────────────┐
                      │        DATA ENGINEERING LAYER         │
                      │  Sync / Clean / Normalize / Resample  │
                      │  Feature Extraction / Event Segment   │
                      └──────────────┬───────────────────────┘
                                     │
                   ┌─────────────────┼──────────────────┐
                   │                 │                   │
                   ▼                 ▼                   ▼
        ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
        │ SoLEXS Detector  │ │ HEL1OS Det.  │ │  Feature Store   │
        │ (Wavelet + ML)   │ │ (Wavelet+ML) │ │ (Physics + Temp) │
        └────────┬─────────┘ └──────┬───────┘ └────────┬─────────┘
                 │                  │                   │
                 └────────┬─────────┘                   │
                          │                             │
                          ▼                             │
               ┌──────────────────┐                     │
               │ Master Flare Cat.│                     │
               │ (Merge + Dedup)  │                     │
               └────────┬─────────┘                     │
                        │                               │
                        ▼                               │
               ┌────────────────────────────────────────┘
               │                    │
               ▼                    ▼
     ┌──────────────────┐  ┌──────────────────┐
     │  NOWCASTING      │  │  FORECASTING     │
     │  (Detect + Class)│  │  (Predict N min) │
     │  LSTM / Bi-LSTM  │  │  TFT / Informer  │
     └────────┬─────────┘  └────────┬─────────┘
              │                     │
              └──────────┬──────────┘
                         ▼
               ┌────────────────────┐
               │   VISUAL DASHBOARD │
               │  (Streamlit/Gradio)│
               └────────────────────┘
```

### Data Flow Summary

```
Raw Data → Preprocessing → Detection → Catalog Merge → Feature Engineering → Nowcasting/Forecasting → Dashboard
```

---

## 7. Stage 1: Data Engineering

### 5A. SoLEXS Preprocessing

1. **Load Level-1 data** (time series of soft X-ray flux)
2. **Handle missing timestamps** — interpolate if gap < threshold, mark as NaN otherwise
3. **Remove non-physical values** (negative flux, instrument saturation flags)
4. **Normalize flux** — log-transform (flux spans many orders of magnitude during flares)
5. **Estimate baseline** — use rolling median with a wide window (hours) to get quiescent solar background
6. **Compute excess flux** — `flux - baseline` for flare detection
7. **Apply low-pass filter** — remove high-frequency instrumental noise while preserving flare rise/decay

### 5B. HEL1OS Preprocessing

Same as SoLEXS but with attention to:
- Hard X-ray data is spikier (shorter bursts)
- Use adaptive baseline estimation (shorter window for HEL1OS)
- More aggressive denoising may be needed

### 5C. Time Synchronization

- SoLEXS and HEL1OS may have different sampling rates
- Use **cubic spline interpolation** to resample both to a common time grid
- Recommended: **1-second resolution** for detection, **1-minute resolution** for forecasting

### 5D. Noise Reduction Strategy

| Method | When to Use |
|---|---|
| **Moving average** | Quick smoothing, removes high-frequency noise |
| **Savitzky-Golay filter** | Preserves peak shape better than moving average |
| **Wavelet denoising** | Best for non-stationary signals (flare + noise have different frequency content) |
| **Median filter** | Excellent for removing impulse noise/spikes |

### 5E. Spike Handling

- **Threshold spike detection**: If a single point exceeds its neighbors by >5σ, flag as spike
- **Compare with GOES**: If SoLEXS/HEL1OS spikes but GOES doesn't, likely noise
- **Temporal consistency**: A real flare lasts minutes, a spike lasts seconds

### 5F. Resampling Strategy

| Use Case | Resolution | Reason |
|---|---|---|
| Detection | Native (1s) | Preserve fast rise times |
| Classification | 1-min | Smooth noise, capture shape |
| Forecasting | 5-min | Reduce computation, align with SHARP cadence |

### 5G. Feature Extraction from Raw Flux

| Feature | Description |
|---|---|
| **Peak flux** | Maximum value in window |
| **Rise time** | Time from baseline to 50% of peak |
| **Decay time** | Time from peak to 50% of baseline |
| **Duration** | Total time above threshold |
| **Total fluence** | Area under flux curve |
| **Rise rate (dF/dt)** | Speed of energy release |
| **Pre-flare quiescent level** | Baseline just before event |

---

## 8. Stage 2: Detection Pipeline

### 6A. Independent Detectors

We build two separate detectors — one for SoLEXS, one for HEL1OS — then merge results.

### 6B. Detection Methods Compared

| Method | Pros | Cons |
|---|---|---|
| **Threshold-based** (flux > fixed threshold) | Simple, fast | Misses weak flares, triggers on noise, threshold tuning is fragile |
| **Statistical anomaly** (Z-score, MAD) | Data-adaptive | Assumes Gaussian noise, struggles with variable solar background |
| **Wavelet-based** ⭐ | Multi-resolution, excellent for flare-like signals, naturally separates noise from signal | More complex to implement, needs parameter tuning |
| **ML-based** (trained classifier on windows) | Can learn complex patterns | Needs labeled data, computationally expensive |

### 6C. Recommended: Wavelet-Based Detection (Primary)

**Why Wavelets?**

Solar flare signals have structure at multiple time scales:
- Rise phase: high-frequency (fast increase)
- Peak: medium-frequency
- Decay phase: low-frequency (slow decrease)
- Noise: high-frequency, low amplitude

Wavelet transform decomposes the signal into different frequency bands, letting us detect flares in the bands where they are strongest.

**Architecture:**

```
Input Flux → CWT/DWT → Coefficient Thresholding → 
  → Reconstruct → Peak Detection → Event Extraction
```

- Use **Continuous Wavelet Transform (CWT)** with **Morlet wavelet** (best matches flare shape)
- Identify time-frequency regions where coefficients exceed noise floor
- Group adjacent detections into events
- Use adaptive noise floor estimation (per wavelet scale)

**Output per detection:**
- Start time (when flux first exceeds threshold)
- Peak time (maximum flux)
- End time (when flux returns below threshold)
- Duration (end - start)
- Peak flux value

### 6D. ML-Based Detection (Secondary/Backup)

Use a **1D CNN** on short windows of flux:
- Input: 60-second window of normalized flux
- Output: Flare / No Flare
- Training labels from GOES catalog
- Use as cross-validation of wavelet detections

---

## 9. Stage 3: Classification Pipeline

### 7A. Task

Given a detected flare event, classify it into:
- **B** (weakest)
- **C**
- **M**
- **X** (strongest, rarest)

### 7B. Feature Set for Classification

| Feature Type | Examples |
|---|---|
| **Peak flux** | Maximum flux during event |
| **Duration** | How long the flare lasted |
| **Rise rate** | dF/dt (how fast energy was released) |
| **Total fluence** | Area under the curve |
| **Decay time** | How long it took to dissipate |
| **Wavelet energy** | Energy in different frequency bands |

### 7C. Models Compared

| Model | Why Consider | Why Skip/Use |
|---|---|---|
| **Random Forest** | Robust, handles mixed features, interpretable | ⭐ **Recommended** — works well on tabular flare features |
| **XGBoost** | Faster, handles class imbalance better | Use if RF underperforms |
| **LightGBM** | Faster training on large data | Use for larger datasets |
| **CatBoost** | Handles categorical features natively | Less needed here (mostly numeric) |

### 7D. Handling Class Imbalance

X-class flares are rare (~1-5% of events). Use:
- **SMOTE** (oversample minority classes)
- **Class weights** in loss function
- **Stratified sampling** in train/test splits

---

## 10. Stage 4: Master Flare Catalog

### 8A. Purpose

Merge SoLEXS detections and HEL1OS detections into a single unified catalog with a unique ID per flare event.

### 8B. Matching Strategy

Two detections (one from SoLEXS, one from HEL1OS) are considered the **same flare** if:

1. **Temporal overlap** — their time windows overlap by >50%
2. **Peak proximity** — peaks are within 5 minutes of each other
3. **Duration consistency** — durations differ by <2x

### 8C. Merge Rules

| Scenario | Action |
|---|---|
| Both detect, agree | Confidence = High. Use SoLEXS timing (soft X-ray is standard for GOES comparison) |
| SoLEXS detects, HEL1OS misses | Lower confidence. Could be weak event below HEL1OS sensitivity |
| HEL1OS detects, SoLEXS misses | Could be hard X-ray only event. Flag for review |
| Both detect, disagree | Investigate. If timing is very different, could be two separate events |

### 8D. Catalog Schema

| Column | Description |
|---|---|
| Master_Flare_ID | Unique identifier |
| Start_Time | Merged start |
| Peak_Time | Merged peak |
| End_Time | Merged end |
| Duration | Merged duration |
| Peak_Flux_SoLEXS | Peak from SoLEXS |
| Peak_Flux_HEL1OS | Peak from HEL1OS |
| Flare_Class | B/C/M/X (from classification model) |
| Confidence | High / Medium / Low |
| GOES_Confirmed | Whether matched to GOES catalog |

---

## 11. Stage 5: Physics-Informed Feature Engineering

### 9A. Why Physics Matters

Solar flares are caused by magnetic reconnection in active regions. Pure flux time series models may miss precursors. Adding physics-based features from SHARP and sunspot data provides the "why" behind flares.

### 9B. Temporal Features (from flux data)

| Feature | Formula | Physical Meaning |
|---|---|---|
| Flux value | F(t) | Current emission level |
| Lag-1 flux | F(t-1) | Short-term memory |
| Moving avg (5 min) | MA_5 | Smooth trend |
| Moving avg (30 min) | MA_30 | Medium-term trend |
| Flux gradient | dF/dt | Rate of energy release |
| Flux acceleration | d²F/dt² | Change in energy release rate |
| Normalized excess | (F - baseline)/baseline | Relative brightness |

### 9C. Physics Features (from SHARP)

| Feature | Formula | Physical Meaning |
|---|---|---|
| **Magnetic Instability Index** | USFLUX × TOTUSJH | Combines total magnetic flux with current helicity — high values indicate unstable active regions prone to flaring |
| **Energy Accumulation Index** | ∫F(t)dt over rolling window | Integrated flux as proxy for stored magnetic energy |
| **Free Magnetic Energy Proxy** | TOTPOT - TOTBSQ | Excess energy available for flare (potential field vs observed field) |
| **Magnetic Twist** | TOTUSJZ / TOTUSJH | Measure of how twisted the magnetic field lines are |
| **R_VALUE** (raw) | R_VALUE | NOAA's flare forecast parameter — probability of M/X flare in next 24h |
| **Solar Cycle Phase** | Sunspot number | Global activity level indicator |

### 9D. Physical Motivation for Each Feature

1. **USFLUX × TOTUSJH**: Magnetic instability is a known precursor — regions with high flux AND high twist are most likely to flare (Leka & Barnes 2007)
2. **Rolling integrated flux**: Energy must build up before a flare (energy storage phase). Integrated X-ray flux is a proxy for this
3. **TOTPOT - TOTBSQ**: The difference between potential (lowest energy) field and observed field gives free magnetic energy available for flares
4. **dF/dt**: Rapid rise in flux indicates instability onset
5. **Sunspot number**: Solar cycle modulates flare frequency — more sunspots = more flare potential

### 9E. Feature Windows

| Feature Group | Window |
|---|---|
| Flux features | 1 min, 5 min, 30 min |
| SHARP features | Latest available (hourly typically) |
| Sunspot number | Daily |
| Gradient features | 1 min, 5 min |

---

## 12. Stage 6: Nowcasting System

### 10A. What Nowcasting Means

Nowcasting answers: **"Is there a flare happening RIGHT NOW?"**

### 10B. Input

- Current SoLEXS flux (last 5 minutes)
- Current HEL1OS flux (last 5 minutes)
- Recent trend (dF/dt over last 30 seconds)
- Latest detections from wavelet pipeline

### 10C. Output

| Field | Description |
|---|---|
| Flare status | Ongoing / Not ongoing |
| Flare class | B/C/M/X |
| Confidence | % confidence |
| Time since onset | Seconds since detection |
| Peak estimate | Predicted peak flux (updates as flare evolves) |

### 10D. Architecture

```
                      ┌─────────────────────┐
                      │  Wavelet Detector    │
                      │  (Real-time stream)  │
                      └──────────┬──────────┘
                                 │
                      ┌──────────▼──────────┐
                      │   Threshold Check    │
                      │   (Above noise?)     │
                      └──────────┬──────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
     ┌────────────────┐ ┌──────────────┐ ┌────────────────┐
     │ SoLEXS Trigger │ │ Both Trigger │ │ HEL1OS Trigger │
     └────────────────┘ └──────────────┘ └────────────────┘
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                      ┌─────────────────────┐
                      │  Classification     │
                      │  (Random Forest)    │
                      └──────────┬──────────┘
                                 ▼
                      ┌─────────────────────┐
                      │  Alert Generation   │
                      └─────────────────────┘
```

### 10E. Real-time Engine

- Process data in sliding windows (5-minute buffer)
- Update every 1 second
- Emit event when flux exceeds dynamic threshold for >3 consecutive seconds
- Update classification as more data arrives

---

## 13. Stage 7: Forecasting System

### 11A. What Forecasting Means

Forecasting answers: **"Will there be a flare in the next N minutes?"**

### 11B. Forecast Horizons

| Horizon | Use Case |
|---|---|
| 15 min | Immediate warning |
| 30 min | Short-term operations |
| 60 min | Medium-term planning |

### 11C. Input Features

- Last 60 minutes of SoLEXS flux
- Last 60 minutes of HEL1OS flux
- Latest SHARP parameters
- Current sunspot number
- Derived physics features (instability index, energy accumulation)
- Past flare history from master catalog

### 11D. Output

```
Forecast at time T:
  P(Flare in 15 min): 0.75
    ├─ P(B): 0.30
    ├─ P(C): 0.35
    ├─ P(M): 0.08
    └─ P(X): 0.02
  P(Flare in 30 min): 0.65
    ├─ ...
  P(Flare in 60 min): 0.50
    ├─ ...
```

### 11E. Problem Formulation

**Type:** Multi-class, multi-horizon probability forecasting

**Approach 1 (Recommended):** Train separate binary classifiers per class + per horizon, then combine probabilities.
- 12 models: 3 horizons × 4 classes
- Plus 1 overall flare detector per horizon

**Approach 2 (Temporal model):** Single model that outputs all probabilities at once
- Better correlation between classes and horizons
- More complex to train

---

## 14. Model Benchmarking Strategy

### 12A. Use PyCaret for Initial Screening

PyCaret is an automated machine learning library. We use it because:
- Quickly compares 10+ models with default settings
- Provides standardized metrics
- Identifies top 3-5 candidate models fast
- Frees us to focus on feature engineering, not hyperparameter tuning at this stage

### 12B. Benchmarking Pipeline (in PyCaret)

1. Set up classification experiment: `setup(data, target='flare_class', ignore_low_variance=True)`
2. Compare models: `compare_models()`
3. Models compared:
   - Logistic Regression
   - Random Forest (RF)
   - Extra Trees (ET)
   - XGBoost
   - LightGBM
   - CatBoost
   - Gradient Boosting
   - AdaBoost
   - Naive Bayes
   - K-Neighbors
4. Generate ranking table

### 12C. Metrics to Prioritize

| Metric | Why |
|---|---|
| **Recall (TPR)** | Missing a flare is worse than false alarm |
| **Precision** | Too many false alarms erode trust |
| **F1 Score** | Balance of precision and recall |
| **PR-AUC** | Best for imbalanced classes (few X flares) |
| **ROC-AUC** | Overall separability |

### 12D. Expected Outcome

Based on prior solar physics ML literature:
- **CatBoost** or **XGBoost** typically win on tabular features
- **Random Forest** is close second with better interpretability
- Deep models (next sections) outperform on temporal/sequential data

---

## 15. Deep Learning Experiments

### 13A. Why Deep Learning?

Tree-based models work well on static features but struggle with:
- Temporal dependencies (precursor patterns before flares)
- Variable-length sequences
- Multi-variate time series (flux + SHARP all at once)

### 13B. Models to Evaluate

#### LSTM (Long Short-Term Memory)

| Aspect | Detail |
|---|---|
| **Strength** | Excellent for sequential data, handles long-range dependencies, proven in time series forecasting |
| **Weakness** | Computationally expensive, needs lots of data, can overfit on small datasets |
| **Use Case** | Nowcasting (short sequences of recent flux) |

#### GRU (Gated Recurrent Unit)

| Aspect | Detail |
|---|---|
| **Strength** | Simpler than LSTM, fewer parameters, trains faster, similar performance |
| **Weakness** | Slightly less expressive for very long sequences |
| **Use Case** | When we need faster training than LSTM |

#### Bi-LSTM (Bidirectional LSTM)

| Aspect | Detail |
|---|---|
| **Strength** | Looks at past AND future context — powerful for classification of already-detected events |
| **Weakness** | Cannot use for real-time forecasting (needs future data), slower to train |
| **Use Case** | Classification of completed flares in master catalog |

### 13C. Recommended DL Baseline

**LSTM for nowcasting, Bi-LSTM for post-event classification.**

```
Input: (batch, sequence_length, num_features)
        ↓
    LSTM(128) → Dropout(0.3)
        ↓
    LSTM(64) → Dropout(0.3)
        ↓
    Dense(32, ReLU)
        ↓
    Dense(num_classes, Softmax)
```

---

## 16. Advanced Temporal Models

### 14A. TCN (Temporal Convolutional Network)

**Why TCN over LSTM?**
- Parallel computation (not sequential like LSTM) → faster training
- Dilated convolutions capture long-range dependencies without vanishing gradients
- Often achieves better results on time series benchmarks (Bai et al. 2018, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling")

```
Input
  ↓
Conv1D (dilation=1) → BatchNorm → ReLU → Dropout
  ↓
Conv1D (dilation=2) → BatchNorm → ReLU → Dropout
  ↓
Conv1D (dilation=4) → BatchNorm → ReLU → Dropout
  ↓
Conv1D (dilation=8) → BatchNorm → ReLU → Dropout
  ↓
Residual connection (every 2 layers)
  ↓
Global Average Pooling → Dense → Output
```

### 14B. TCN + Attention

Add an **attention mechanism** on top of TCN to:
- Focus on important time steps (e.g., rising phase of flux)
- Highlight precursor signatures before flare onset
- Provide interpretability (which time points matter most)

**Architecture:**

```
Input → TCN layers → Attention(Query=TCN_out, Key=TCN_out, Value=TCN_out)
  → Weighted sum → Dense → Output
```

---

## 17. Transformer-Based Models

### 15A. Transformer Encoder

Standard transformer encoder applied to time series:
- Uses self-attention to capture dependencies across all time steps
- Pros: Very expressive, captures long-range dependencies
- Cons: Needs lots of data, computationally expensive, positional encoding needed

### 15B. Informer (Zhou et al. 2021)

**Why Informer?**
- Designed specifically for **long sequence time-series forecasting**
- **ProbSparse attention** reduces computation from O(L²) to O(L log L)
- Handles long input sequences + long output horizons
- Built for solar flare-like problems: long lookback, variable-length forecast

### 15C. Temporal Fusion Transformer (TFT) ⭐

**Why TFT is the best fit for this problem:**

| Feature | How It Helps |
|---|---|
| **Static covariate encoder** | Incorporates sunspot number (global context) |
| **Past inputs** | Uses recent flux history (30-60 min) |
| **Known future inputs** | None for our problem (we forecast), but TFT handles this gracefully |
| **Variable selection network** | Automatically weights SHARP features + flux features appropriately |
| **Temporal self-attention** | Learns which past time steps are precursors |
| **Quantile outputs** | Can output prediction intervals (uncertainty estimates!) |
| **Explainability** | Attention weights show which features/time steps drove the prediction |

**Architecture (simplified):**

```
Input Features (Flux + SHARP + Sunspot + Time)
        ↓
Variable Selection Network
  ┌─────────────────────┐
  │ Weights which       │
  │ features matter     │
  └─────────────────────┘
        ↓
LSTM Encoder (past sequence)
        ↓
Multi-Head Attention + Add & Norm
        ↓
LSTM Decoder (future)
        ↓
Quantile Outputs
  ├─ P10 (lower bound)
  ├─ P50 (median prediction)
  └─ P90 (upper bound)
```

**TFT is our top candidate for the forecasting model.**

---

## 18. Physics-Informed Learning (PINN)

### 16A. What is PINN?

A Physics-Informed Neural Network adds a **physics loss term** to the normal prediction loss. The physics loss penalizes predictions that violate known physical laws.

### 16B. Is Pure PINN Suitable?

**No.** A pure PINN (where the entire model is constrained by PDEs) is NOT suitable because:
- Solar flare physics is not fully described by closed-form PDEs
- Magnetic reconnection is a complex, multi-scale process
- We lack a universal governing equation for flare occurrence

### 16C. Is Hybrid PINN Better?

**Yes.** A hybrid approach adds physics-inspired regularization:
- Push the model to prefer solutions that are consistent with known physics
- Don't hard-constrain, just softly guide

### 16D. Proposed Physics Constraints

These constraints are supported by solar physics literature:

1. **Energy buildup constraint**: A flare cannot occur without prior energy accumulation
   - Loss: Penalize predictions of large flares when integrated flux (energy proxy) is low

2. **Magnetic instability constraint**: Flare probability should increase with SHARP instability index
   - Loss: Penalize situations where model predicts no flare but SHARP index is very high

3. **Temporal smoothness**: Flux evolution should be smooth (flares don't appear/disappear instantly)
   - Loss: Penalize unrealistic dF/dt values

### 16E. Architecture: TFT + Physics Loss

```
TFT Forward Pass → Prediction (y_pred)
       ↓
Prediction Loss = CrossEntropy(y_pred, y_true)
       +
Physics Loss = λ₁ * EnergyBuildupLoss + λ₂ * InstabilityConsistencyLoss + λ₃ * SmoothnessLoss
       ↓
Total Loss = Prediction Loss + Physics Loss
```

### 16F. Architecture: Transformer Encoder + Physics Loss

Same approach as above but with transformer encoder replacing TFT.

### 16G. Important Caveat

We do **not** invent physics equations. Only use constraints that can be cited from literature:
- Leka & Barnes (2007) — magnetic complexity → flare probability
- Falconer et al. (2008) — free magnetic energy → flare potential
- Schrijver (2007) — non-potential magnetic field → flare likelihood

---

## 19. Advanced Techniques from Research Papers

This section surveys the latest published research on solar flare forecasting with deep learning. These techniques are evaluated for potential integration into the system.

---

### 19A. SolarFlareNet — Transformer + SHARP Time Series (2023)

**Source:** Abduallah et al., *Scientific Reports* 13, 13665 (2023). Extended with LIME/ALE interpretability (Gazula et al., FLAIRS 2024).

**What it is:** A transformer-based framework predicting ≥C, ≥M, or ≥M5.0 class flares within 24-72 hours using SHARP time series.

**Key findings:**
- Operates on SHARP time series alone (no images needed)
- Separate transformer per flare class head
- Extended to probabilistic forecasts via calibration
- Operational system running at NJIT

**Integration:** Adapt same transformer-on-SHARP-time-series approach. Use as benchmark.

**Verdict:** ✅ **Adopt** — proven operational system with same data modality

---

### 19B. Moirai2 — Foundational Transformer for Time Series (2025)

**Source:** Riggi et al., *Astronomy & Computing* 55, 101042 (2026). arXiv:2510.23400.

**What it is:** A pretrained foundational time-series transformer, fine-tuned on GOES X-ray flux for flare forecasting.

**Key findings:**
- Achieves **TSS ≈ 0.74** using irradiance-based temporal evolution alone
- Outperforms image-based models (SigLIP2 TSS 0.60, VideoMAE TSS 0.65)
- Pretrained on LOTSA (large time-series corpus) — patch-based processing
- Time-series-only modality outperforms image/video modalities

**Integration:** Fine-tune Moirai2 as alternative to training TFT from scratch. Lower risk, less data needed.

**Verdict:** ✅ **Strong candidate** — pretrained foundation model reduces training data requirements

---

### 19C. Flare-PINN — Weak-Form Physics-Informed Learning (2025)

**Source:** GitHub: Flare-PINN/Flare-PINN (MIT License)

**What it is:** Hybrid PINN combining MHD constraints with deep learning for operational flare forecasting.

**Key findings:**
- **Weak-form PINN** outperforms strong-form PINN (+0.089 TSS at 12h, +0.038 at 24h)
- Physics improves TSS by **+0.052 at 24h** over no-physics ablation (p=0.010)
- Achieves **TSS 0.790 ± 0.007** at 24h horizon
- Distance-to-corner (D2C) thresholding for optimal ROC performance

**Integration:** Use weak-form approach for our physics-informed TFT — add MHD constraints as soft loss terms.

**Verdict:** ✅ **Adopt weak-form PINN approach** — concrete implementation blueprint for physics regularization

---

### 19D. Decomposition-LSTM (DLSTM) with Sliding Windows (2025)

**Source:** Hassani et al., *ApJS* 279, 27 (2025). arXiv:2507.05313.

**What it is:** LSTM combined with time series decomposition (trend + seasonal) and sliding window pattern recognition using GOES flux.

**Key findings:**
- DLSTM + ensemble achieves **Recall 0.95, AUC 0.87, TSS 0.74**
- Decomposition isolates noise from signal — critical for flare data
- Regularized (3h interval) time series outperforms irregular
- Sliding window detects temporal quasi-patterns across solar cycles

**Integration:** Add STL decomposition before TFT/LSTM input. Decompose flux into trend + seasonal + residual components.

**Verdict:** ✅ **Adopt decomposition preprocessing** — improves SNR for all downstream models

---

### 19E. N-HiTS — Neural Hierarchical Interpolation (2023)

**Source:** Challu et al., *AAAI* 2023. arXiv:2201.12886.

**What it is:** Neural architecture using hierarchical interpolation and multi-rate sampling for long-horizon forecasting.

**Key findings:**
- **20% average accuracy improvement** over Transformer baselines
- **50x reduction** in computation time
- Built-in interpretability via basis expansion
- N-HiTS extends N-BEATS with covariate support and probabilistic outputs

**Integration:** Lightweight alternative to TFT for forecasting horizons ≥30 min. Benchmark against TFT.

**Verdict:** ✅ **Promising lightweight alternative** — especially for resource-constrained deployment

---

### 19F. Causal Attention Model (2024)

**Source:** Zheng et al., *ApJS* 274, 38 (2024).

**What it is:** Deep learning model with causal attention module that disentangles causal features from confounders.

**Key findings:**
- Improves TSS by 4-8% over non-causal baselines
- Adaptive data split handles class imbalance dynamically
- Focuses on causal precursors rather than spurious correlations

**Integration:** Could be integrated into TFT's attention mechanism to reduce false alarms.

**Verdict:** ⚠️ **Consider for Phase 5** — high complexity, promising for FPR reduction

---

### 19G. CNN-SE / CNN-CBAM / CNN-ECA Attention (2024)

**Source:** Yan et al., *Astrophysics and Space Science* 369, 110 (2024).

**What it is:** CNN models augmented with channel attention mechanisms for flare forecasting from magnetograms.

**Key findings:**
- CNN-SE achieves **TSS 0.984 for ≥C-class**, BSS 0.939 in real-time operation
- CNN-ECA and ViT show best Recall for ≥M-class (0.799 and 0.855)
- Real-time operational system since May 2023

**Integration:** Add SE/ECA attention blocks to TCN baseline as lightweight enhancement.

**Verdict:** ✅ **Add SE/ECA attention to TCN** — low computational cost, proven improvement

---

### 19H. FSPT — Flare Set-Prediction Transformer (2025)

**Source:** MDPI *Universe* 11(6), 174 (2025).

**What it is:** Transformer adapted from DETR (object detection) that directly forecasts a variable-sized set of flare events.

**Key findings:**
- Predicts set of events (start, peak, end, class) end-to-end
- Bipartite matching loss handles variable number of events
- Paradigm shift from binary classification to set prediction

**Integration:** Could combine detection + forecasting in one model. Novel but unproven.

**Verdict:** ⚠️ **Research-stage** — monitor for future adoption

---

### 19I. PINT — Physics-Informed Neural Time Series (2025)

**Source:** Park et al., arXiv:2502.04018 (2025).

**What it is:** Framework integrating physical constraints (e.g., SHM equation) into neural time series models.

**Key findings:**
- Physics constraints reduce data requirements
- Improves long-term inference stability
- Applies to any dynamics with known periodicity

**Integration:** Add solar cycle (11-year) periodicity constraint for long-horizon forecasts.

**Verdict:** ✅ **Consider periodic physics constraints** for solar cycle modulation

---

### 19J. 3DTCN — 3D Temporal Convolutional Networks (2025)

**Source:** Guesmi et al., "EoFTCNets", Research Square (2025).

**What it is:** Extends TCN to 3D for spatio-temporal analysis of active region patches.

**Key findings:**
- Captures spatial and temporal correlations simultaneously
- Separate predictor modules per flare class
- Nowcasting system (not just forecasting)

**Integration:** If we add magnetogram data in future phases.

**Verdict:** ❌ **Future consideration** — requires image data we don't currently have

---

### 19K. Comparison Summary

| Technique | Year | Input | Key Metric | Operational? | Priority |
|---|---|---|---|---|---|
| **SolarFlareNet** (Transformer) | 2023 | SHARP TS | TSS >0.80 | ✅ Yes | **High** |
| **Flare-PINN** (Weak-Form) | 2025 | SHARP TS | TSS 0.79 | ❌ Research | **High** |
| **DLSTM + Decomposition** | 2025 | GOES flux | TSS 0.74 | ❌ Research | **Medium** |
| **N-HiTS** (Hierarchical) | 2023 | Univariate TS | +20% vs Transformers | ❌ Research | **Medium** |
| **CNN-SE/ECA** | 2024 | Magnetograms | TSS 0.984 (C) | ✅ Yes | **Medium** |
| **Moirai2** (Foundation) | 2025 | GOES flux | TSS 0.74 | ❌ Research | **Medium** |
| **Causal Attention** | 2024 | Magnetograms | TSS +4-8% | ❌ Research | **Low** |
| **FSPT** (Set Prediction) | 2025 | Varied | Reported | ❌ Research | **Low** |
| **PINT** (Physics NTS) | 2025 | Varied | Stable | ❌ Research | **Low** |
| **3DTCN** | 2025 | Mag. video | Reported | ❌ Research | **Low** |

### 19L. Research Integration Plan

```
Phase 2-3 (Core Pipeline):
  ├── TFT (primary forecasting model)
  ├── N-HiTS (lightweight alternative benchmark)
  └── DLSTM decomposition preprocessing (all models)

Phase 4 (Enhanced):
  ├── Flare-PINN weak-form physics loss
  ├── CNN-SE/ECA attention on TCN baseline
  └── SolarFlareNet-style multi-head per-class outputs

Phase 5 (Advanced):
  ├── Causal attention for false positive rate reduction
  ├── PINT periodic physics constraints (solar cycle)
  └── Moirai2 fine-tuning experiment
```

---

## 20. Final Recommended Architecture

### 20A. Summary of Candidates

| Rank | Model | Accuracy | Scientific Validity | Interpretability | Feasibility |
|---|---|---|---|---|---|
| A | TCN + Attention | High | Medium | Medium | High |
| B | Informer | High | Medium | Low | Medium |
| C | **Temporal Fusion Transformer** ⭐ | **Highest** | **High** | **High** | **High** |
| D | Physics-Informed TFT | Highest | Highest | High | Medium |
| E | Physics-Informed Transformer Encoder | High | High | Medium | Medium |

### 20B. Final Recommendation: Temporal Fusion Transformer (TFT)

**Why TFT wins:**

1. **Best fit for problem structure**: TFT is designed for time series with static features (sunspot number), past observed inputs (flux history), and optional future known inputs
2. **Variable selection network**: Automatically learns to weight SHARP features, flux features, and sunspot data appropriately — no manual feature importance analysis needed
3. **Interpretable attention**: We can see exactly which time steps and features drove each forecast — critical for scientific validation
4. **Quantile regression**: Gives us uncertainty estimates (P10/P50/P90) — essential for operational forecasting
5. **Proven in similar problems**: TFT has been successfully applied to energy forecasting, demand prediction, and climate time series
6. **Implementation available**: PyTorch Forecasting library has a well-maintained TFT implementation

### 20C. Complete Final Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     DATA ENGINEERING                          │
│  ┌────────┐  ┌────────┐  ┌──────┐  ┌──────┐  ┌─────────┐   │
│  │SoLEXS  │  │HEL1OS  │  │GOES  │  │SHARP │  │Sunspot  │   │
│  │Preproc │  │Preproc │  │Catalog│  │Params│  │Numbers  │   │
│  └───┬────┘  └───┬────┘  └──┬───┘  └──┬───┘  └────┬────┘   │
│      └─────┬─────┘         │         │           │         │
│            ▼               │         │           │         │
│      ┌──────────┐          │         │           │         │
│      │ Synced   │          │         │           │         │
│      │ Time     │          │         │           │         │
│      │ Series   │          │         │           │         │
│      └────┬─────┘          │         │           │         │
└───────────┼────────────────┼─────────┼───────────┼─────────┘
            │                │         │           │
            ▼                ▼         ▼           ▼
     DETECTION          FEATURE STORE (Stage 5)
 ┌────────────────┐
 │ Wavelet-based  │     ┌──────────────────────────────┐
 │ Detection      │     │ Temporal: flux, dF/dt, MA    │
 │ (SoLEXS+HEL1OS)│     │ Physics: MII, EAI, TOTPOT   │
 │ + Threshold    │     │ Context: sunspot #, R_VALUE  │
 │ Check          │     └──────────┬───────────────────┘
 └───────┬────────┘                │
         │                         │
         ▼                         ▼
 ┌────────────────┐     ┌──────────────────────────────┐
 │ Master Flare   │     │   FEATURE ALIGNMENT          │
 │ Catalog        │     │   Align all features to      │
 │ (Merged)       │     │   common time grid           │
 └────────────────┘     └──────────┬───────────────────┘
                                    │
                                    ▼
 ┌────────────────────────────────────────────────────────┐
 │                    TFT MODEL                            │
 │  ┌──────────────────────────────────────────────────┐  │
 │  │ Encoder: Process past sequence (60 min lookback) │  │
 │  │  - Flux from SoLEXS & HEL1OS                     │  │
 │  │  - SHARP parameters (latest available)           │  │
 │  │  - Physics features (instability, energy)        │  │
 │  │  - Sunspot number                                │  │
 │  ├──────────────────────────────────────────────────┤  │
 │  │ Decoder: Forecast future (15/30/60 min ahead)    │  │
 │  │  - Output: P(flare), P(B), P(C), P(M), P(X)     │  │
 │  │  - Quantiles: P10, P50, P90 (uncertainty)        │  │
 │  ├──────────────────────────────────────────────────┤  │
 │  │ Optional: Physics loss regularization            │  │
 │  │  - Energy buildup constraint                     │  │
 │  │  - Magnetic instability consistency              │  │
 │  └──────────────────────────────────────────────────┘  │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │                    OUTPUT LAYER                         │
 │  ┌────────┐ ┌────────┐ ┌──────────┐ ┌──────────────┐  │
 │  │Nowcast │ │Forecast│ │Master    │ │Confidence    │  │
 │  │Status  │ │Probs   │ │Catalog   │ │Scores        │  │
 │  └────────┘ └────────┘ └──────────┘ └──────────────┘  │
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │                  VISUAL DASHBOARD                       │
 │  Streamlit/Gradio + Plotly + real-time updates         │
 └────────────────────────────────────────────────────────┘
```

---

## 21. Evaluation Strategy

### 19A. Data Split Strategy

| Split | Proportion | Years | Purpose |
|---|---|---|---|
| Training | 60% | Earliest | Model training |
| Validation | 20% | Middle | Hyperparameter tuning |
| Testing | 20% | Latest | Final evaluation (temporal holdout) |

**Important:** Split by time, NOT randomly. Random split leaks future information into training.

### 19B. Detection Metrics

| Metric | Formula | Why |
|---|---|---|
| **Precision** | TP / (TP + FP) | How many detections were real flares |
| **Recall** | TP / (TP + FN) | How many real flares were detected |
| **F1** | 2×P×R/(P+R) | Overall detection quality |
| **TPR** | Same as Recall | True Positive Rate |
| **FPR** | FP / (FP + TN) | False alarm rate — critical to minimize |
| **Temporal accuracy** | | How close are detected times to actual |

### 19C. Classification Metrics

| Metric | Why |
|---|---|
| **Macro F1** | Average F1 per class (treats rare X-class fairly) |
| **Weighted F1** | Average F1 weighted by class frequency |
| **Confusion Matrix** | Where are misclassifications? (e.g., M called as X vs M called as C) |

### 19D. Forecasting Metrics

| Metric | Why |
|---|---|
| **PR-AUC** | Precision-Recall AUC — best for imbalanced flare classes |
| **ROC-AUC** | Separability of flare vs no-flare |
| **Brier Score** | Mean squared error of probability predictions (lower is better) |
| **Lead Time** | How early before onset was the alarm raised? |

### 19E. How to Measure Lead Time

Lead time = `(first time model predicts "flare") - (actual flare start time)`

- For a positive prediction: time at which P(flare) first exceeds threshold (e.g., 0.5) before the flare actually starts
- Lead time is only defined for **correct predictions** (true positives)
- Report: **mean lead time**, **median lead time**, **minimum lead time**
- Goal: maximize lead time while maintaining high recall

```
Example:
  Flare starts at T=10:00
  Model first says P(flare)>0.5 at T=09:45
  Lead time = 15 minutes ✓
```

---

## 22. Deployment & Dashboard

### 22A. Dashboard Framework

**Recommended:** Streamlit (Python)

Reasoning:
- Python-native
- Easy to build with data science tools
- Handles real-time updates with st.rerun
- Plotly integration for interactive charts
- Simple to deploy

### 22B. Dashboard Sections

```
┌──────────────────────────────────────────────────────────┐
│              SPACE WEATHER INTELLIGENCE SYSTEM           │
├──────────────────────────────────────────────────────────┤
│ ┌────────────────────┐  ┌──────────────────────────────┐│
│ │ CURRENT STATUS     │  │ NOWCASTING ALERTS            ││
│ │ [●] No Flare      │  │ ┌────────────────────────┐   ││
│ │ - or -             │  │ │ Alert: M-class flare   │   ││
│ │ [🔴] M-Class Flare │  │ │ detected at 10:23 UTC  │   ││
│ │ Active Since: 10:23│  │ │ Confidence: 92%        │   ││
│ │ Duration: 4.2 min  │  │ │ Current phase: Rise    │   ││
│ └────────────────────┘  │ │ Peak estimate: 12 min  │   ││
│                          │ └────────────────────────┘   ││
│ ┌────────────────────┐  └──────────────────────────────┘│
│ │ FORECAST           │                                   │
│ │ Next 15min │ 30min │ 60min                            │
│ │ P(Flare)  0.34  │ 0.52 │ 0.65                        ││
│ │ P(B): 0.10  0.08   0.05                               ││
│ │ P(C): 0.18  0.28   0.35                               ││
│ │ P(M): 0.05  0.14   0.22                               ││
│ │ P(X): 0.01  0.02   0.03                               ││
│ └────────────────────┘                                   │
├──────────────────────────────────────────────────────────┤
│ ┌────────────────────┐  ┌──────────────────────────────┐│
│ │ LIVE FLUX PLOT    │  │ MASTER CATALOG TABLE         ││
│ │ [Interactive chart │  │ │ID│Time│Class│Conf│Source│ ││
│ │  of SoLEXS/HEL1OS] │  │ ──────────────────────────  ││
│ │                    │  │ │01│10:23│ M  │0.92│SoLEXS │ ││
│ │                    │  │ │02│09:15│ C  │0.88│Merged │ ││
│ │                    │  │ └──────────────────────────┘ ││
│ └────────────────────┘  └──────────────────────────────┘│
├──────────────────────────────────────────────────────────┤
│ │ HISTORICAL EVENTS │ Model Confidence │ Performance    │
│ │ [Timeline chart]  │ [Gauge + tooltip]│ [Metrics panel]│
└──────────────────────────────────────────────────────────┘
```

### 22C. API Endpoints

For integration with other systems:

| Endpoint | Method | Returns |
|---|---|---|
| `/api/status` | GET | Current flare status |
| `/api/forecast` | GET | Forecast probabilities for all horizons |
| `/api/catalog` | GET | Master flare catalog |
| `/api/nowcast` | GET | Nowcasting alert details |

### 22D. Real-Time Update Cycle

```
Every 1 second:
  → Read latest SoLEXS/HEL1OS data
  → Run wavelet detection
  → Update nowcast status
  → If flare detected, classify
  → Update dashboard
  
Every 1 minute:
  → Run forecast model
  → Update forecast probabilities
  → Update dashboard
  
Every 6 hours:
  → Fetch latest SHARP data
  → Update physics features
  → Re-run forecast if needed
```

---

## 23. Ablation Study Plan

Ablation studies answer: "Which components actually contribute to performance?"

### 21A. Detection Ablation

| Experiment | What We Remove | What We Learn |
|---|---|---|
| A1 | Wavelet → use only threshold | Does wavelet help? |
| A2 | HEL1OS data | Is HEL1OS necessary or is SoLEXS enough? |
| A3 | SoLEXS data | Is SoLEXS necessary or is HEL1OS enough? |
| A4 | ML validation | Is wavelet-only sufficient? |

### 21B. Feature Ablation

| Experiment | What We Remove | What We Learn |
|---|---|---|
| B1 | All SHARP features | How much do magnetic parameters help? |
| B2 | All sunspot data | Does global activity context matter? |
| B3 | Physics features only | Are physics-informed features sufficient alone? |
| B4 | Temporal features only | Are raw flux features sufficient alone? |
| B5 | Individual SHARP features | Which magnetic parameter is most important? |

### 21C. Model Ablation

| Experiment | What We Change | What We Learn |
|---|---|---|
| C1 | TFT → Simple LSTM | Does TFT attention help? |
| C2 | TFT → XGBoost (no temporal) | Is temporal modeling necessary? |
| C3 | TFT → TCN + Attention | Is TFT better than TCN? |
| C4 | Remove physics loss | Does physics regularization help? |
| C5 | Remove quantile output | Are uncertainty estimates valuable? |

### 21D. Forecasting Horizon Ablation

| Experiment | What We Change | What We Learn |
|---|---|---|
| D1 | 15-min only | Can we forecast at all? |
| D2 | 60-min only | How far ahead can we forecast? |
| D3 | All horizons together | Does joint training help? |

---

## 24. Research-Grade Experimentation Plan

### 22A. Experiment Tracking

Use **MLflow** for:
- Logging all hyperparameters
- Tracking metrics per experiment
- Model versioning
- Comparing runs

### 22B. Experiment Log Template

| Field | Value |
|---|---|
| Experiment ID | exp_001 |
| Date | YYYY-MM-DD |
| Model | TFT |
| Features | all |
| Horizon | 15, 30, 60 min |
| Loss | CrossEntropy + Physics(λ=0.1) |
| Val PR-AUC | 0.85 |
| Test PR-AUC | 0.82 |
| Lead Time | 12.3 min |
| Notes | Physics loss seemed to improve M-class recall |

### 22C. Statistical Significance

- Run each experiment 3 times with different seeds
- Report mean ± std
- Use paired bootstrap for model comparison

### 22D. Hyperparameter Search

| Model | Tuning Method | Parameters |
|---|---|---|
| TFT | Bayesian (Optuna) | hidden_size, dropout, attention_heads, learning_rate, hidden_continuous_size |
| LSTM | Grid search | num_layers, hidden_size, dropout, learning_rate |
| XGBoost | Bayesian (Optuna) | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

---

## 25. Implementation Roadmap

### Phase 1: Foundation (Week 1)

```
Day 1-2: Setup & Data
  - Clone repo structure
  - Download GOES catalog
  - Download SHARP data
  - Understand SoLEXS/HEL1OS data format
  - Write data loaders

Day 3-4: Preprocessing
  - Implement SoLEXS preprocessing
  - Implement HEL1OS preprocessing
  - Time synchronization
  - Noise reduction
  - Baseline estimation

Day 5-7: Detection (Milestone 1)
  - Implement wavelet-based detection
  - Test on SoLEXS data
  - Test on HEL1OS data
  - Compare with threshold methods
  - Tune detection parameters
```

### Phase 2: Catalog & Classification (Week 2)

```
Day 8-9: Master Catalog
  - Implement merge algorithm
  - Deduplication
  - Create unified catalog
  - Validate against GOES
  - Generate Master Flare IDs

Day 10-11: Classification
  - Extract features for classification
  - PyCaret benchmark
  - Train Random Forest classifier
  - Evaluate confusion matrix
  - Class imbalance handling

Day 12-14: Feature Engineering
  - Compute physics features
  - Temporal features (lags, derivatives)
  - Feature store implementation
  - SHARP feature integration
  - Sunspot feature integration
```

### Phase 3: Nowcasting (Week 3)

```
Day 15-16: Nowcasting Engine
  - Real-time wavelet detector
  - Sliding window processing
  - Alert generation logic
  - Confidence estimation

Day 17-18: Nowcasting Model
  - LSTM for real-time classification
  - Train on detected events
  - Evaluate on held-out data
  - Test with simulated real-time feed
```

### Phase 4: Forecasting (Week 3-4)

```
Day 19-21: Baseline Forecasting
  - PyCaret for initial forecasting benchmark
  - Feature alignment for forecasting
  - Train XGBoost forecast model
  - Evaluate PR-AUC, lead time

Day 22-24: Deep Forecasting
  - Install PyTorch Forecasting
  - Implement TFT model
  - Train with temporal cross-validation
  - Hyperparameter tuning with Optuna
  - Compare with LSTM baseline
```

### Phase 5: Advanced (Week 4)

```
Day 25-26: Physics-Informed
  - Implement physics loss
  - Train TFT + physics loss
  - Ablation: with vs without physics
  - Compare results

Day 27-28: Evaluation & Analysis
  - Full evaluation on test set
  - Lead time analysis
  - Error analysis
  - Ablation studies
  - Statistical significance tests
```

### Phase 6: Deployment (Week 5)

```
Day 29-30: Dashboard
  - Build Streamlit dashboard
  - Real-time flux plots
  - Nowcast alerts
  - Forecast probabilities
  - Master catalog view
  - Historical event browser

Day 31-32: API & Integration
  - REST API endpoints
  - Data pipeline automation
  - Docker containerization
  - Deployment testing

Day 33-35: Final
  - Documentation
  - Demo preparation
  - Final evaluation report
  - README update
```

### Implementation Order (Critical Path)

```
1. Data Engineering
   ↓
2. Detection Pipeline (Wavelet)
   ↓
3. Master Flare Catalog
   ↓
4. Classification Pipeline
   ↓
5. Feature Engineering (Physics + Temporal)
   ↓
6. Nowcasting System
   ↓
7. Forecasting System (TFT)
   ↓
8. Physics-Informed Extension
   ↓
9. Dashboard & Deployment
```

---

## Appendix: Dependencies & Tools

### Core Libraries

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `scipy` | Signal processing, wavelet transforms |
| `pywt` | Wavelet analysis |
| `scikit-learn` | Baseline ML models |
| `xgboost`, `lightgbm`, `catboost` | Gradient boosting models |
| `pycaret` | AutoML benchmarking |
| `pytorch` | Deep learning framework |
| `pytorch-forecasting` | TFT implementation |
| `optuna` | Hyperparameter optimization |
| `mlflow` | Experiment tracking |
| `streamlit` | Dashboard |
| `plotly` | Interactive visualization |
| `fastapi` | API endpoints |

### Data Storage

- **Feature Store**: Parquet files (partitioned by date)
- **Model Registry**: MLflow model registry
- **Catalog**: SQLite database for master flare catalog

---

*This document serves as the complete architecture, research plan, and implementation roadmap for the ISRO Aditya-L1 Solar Flare Nowcasting and Forecasting System.*
