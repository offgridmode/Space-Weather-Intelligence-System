# Space Weather Intelligence System

## ISRO Aditya-L1 Solar Flare Nowcasting & Forecasting System

A research-grade AI system for detecting, classifying, nowcasting, and forecasting solar flares using data from India's Aditya-L1 mission (SoLEXS & HEL1OS payloads), augmented with NOAA GOES catalog, SHARP magnetic parameters, and sunspot data.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Project Goals](#2-project-goals)
3. [Data Sources](#3-data-sources)
4. [System Architecture (Big Picture)](#4-system-architecture-big-picture)
5. [Stage 1: Data Engineering](#5-stage-1-data-engineering)
6. [Stage 2: Detection Pipeline](#6-stage-2-detection-pipeline)
7. [Stage 3: Classification Pipeline](#7-stage-3-classification-pipeline)
8. [Stage 4: Master Flare Catalog](#8-stage-4-master-flare-catalog)
9. [Stage 5: Physics-Informed Feature Engineering](#9-stage-5-physics-informed-feature-engineering)
10. [Stage 6: Nowcasting System](#10-stage-6-nowcasting-system)
11. [Stage 7: Forecasting System](#11-stage-7-forecasting-system)
12. [Model Benchmarking Strategy](#12-model-benchmarking-strategy)
13. [Deep Learning Experiments](#13-deep-learning-experiments)
14. [Advanced Temporal Models](#14-advanced-temporal-models)
15. [Transformer-Based Models](#15-transformer-based-models)
16. [Physics-Informed Learning (PINN)](#16-physics-informed-learning-pinn)
17. [Less Common Research Models](#17-less-common-research-models)
18. [Final Recommended Architecture](#18-final-recommended-architecture)
19. [Evaluation Strategy](#19-evaluation-strategy)
20. [Deployment & Dashboard](#20-deployment--dashboard)
21. [Ablation Study Plan](#21-ablation-study-plan)
22. [Research-Grade Experimentation Plan](#22-research-grade-experimentation-plan)
23. [Implementation Roadmap](#23-implementation-roadmap)

---

## 1. What Is This Project?

Solar flares are sudden, intense bursts of radiation from the Sun. They can disrupt satellites, power grids, and communication systems on Earth. This project builds an **AI-powered system** that:

- **Detects** solar flares as they happen (using SoLEXS and HEL1OS data from Aditya-L1)
- **Classifies** them into B, C, M, X categories (by strength)
- **Nowcasts** (detects in real-time) ongoing flare activity
- **Forecasts** the probability of a flare occurring in the next 15/30/60 minutes

Think of it as a "weather radar" for solar storms — but powered by machine learning and Indian space data.

---

## 2. Project Goals

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

## 3. Data Sources

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

## 4. System Architecture (Big Picture)

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

## 5. Stage 1: Data Engineering

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

## 6. Stage 2: Detection Pipeline

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

## 7. Stage 3: Classification Pipeline

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

## 8. Stage 4: Master Flare Catalog

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

## 9. Stage 5: Physics-Informed Feature Engineering

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

## 10. Stage 6: Nowcasting System

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

## 11. Stage 7: Forecasting System

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

## 12. Model Benchmarking Strategy

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

## 13. Deep Learning Experiments

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

## 14. Advanced Temporal Models

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

## 15. Transformer-Based Models

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

## 16. Physics-Informed Learning (PINN)

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

## 17. Less Common Research Models

### 17A. Neural ODE (Ordinary Differential Equations)

| Aspect | Detail |
|---|---|
| **What it is** | Neural network that models the derivative of the hidden state, not the hidden state itself |
| **Complexity** | High — requires ODE solvers, computationally expensive |
| **Data req.** | Very high — needs dense, continuous time series |
| **Expected benefit** | Could model continuous flux evolution better than discrete-time LSTM |
| **Verdict** | ❌ Not recommended for hackathon timeframe. Too complex, benefit speculative |

### 17B. Graph Neural Networks (GNN)

| Aspect | Detail |
|---|---|
| **What it is** | Neural network on graph-structured data (nodes = active regions, edges = magnetic connectivity) |
| **Complexity** | High — requires defining graph structure between active regions |
| **Data req.** | Requires position/magnetic connectivity data beyond current datasets |
| **Expected benefit** | Could model flare interactions between active regions |
| **Verdict** | ❌ Not recommended. We lack multi-region connectivity data |

### 17C. DeepAR (Amazon)

| Aspect | Detail |
|---|---|
| **What it is** | Autoregressive RNN for probabilistic forecasting |
| **Complexity** | Moderate |
| **Data req.** | Moderate — needs multiple related time series |
| **Expected benefit** | Probabilistic forecasts with uncertainty |
| **Verdict** | ✅ Worth trying if TFT doesn't work. Simpler than TFT, less expressive |

### 17D. Attention-based TCN

| Aspect | Detail |
|---|---|
| **What it is** | TCN with attention mechanism on top |
| **Complexity** | Moderate |
| **Data req.** | Moderate |
| **Expected benefit** | Combines TCN speed with attention interpretability |
| **Verdict** | ✅ Good candidate. Recommended as TCN benchmark |

### 17E. Physics-Informed Transformer

| Aspect | Detail |
|---|---|
| **What it is** | Transformer with physics loss regularization |
| **Complexity** | High |
| **Data req.** | High |
| **Expected benefit** | Combines transformer expressiveness with physical consistency |
| **Verdict** | ✅ Recommended as secondary model. Primary = TFT |

---

## 18. Final Recommended Architecture

### 18A. Summary of Candidates

| Rank | Model | Accuracy | Scientific Validity | Interpretability | Feasibility |
|---|---|---|---|---|---|
| A | TCN + Attention | High | Medium | Medium | High |
| B | Informer | High | Medium | Low | Medium |
| C | **Temporal Fusion Transformer** ⭐ | **Highest** | **High** | **High** | **High** |
| D | Physics-Informed TFT | Highest | Highest | High | Medium |
| E | Physics-Informed Transformer Encoder | High | High | Medium | Medium |

### 18B. Final Recommendation: Temporal Fusion Transformer (TFT)

**Why TFT wins:**

1. **Best fit for problem structure**: TFT is designed for time series with static features (sunspot number), past observed inputs (flux history), and optional future known inputs
2. **Variable selection network**: Automatically learns to weight SHARP features, flux features, and sunspot data appropriately — no manual feature importance analysis needed
3. **Interpretable attention**: We can see exactly which time steps and features drove each forecast — critical for scientific validation
4. **Quantile regression**: Gives us uncertainty estimates (P10/P50/P90) — essential for operational forecasting
5. **Proven in similar problems**: TFT has been successfully applied to energy forecasting, demand prediction, and climate time series
6. **Implementation available**: PyTorch Forecasting library has a well-maintained TFT implementation

### 18C. Complete Final Architecture

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

## 19. Evaluation Strategy

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

## 20. Deployment & Dashboard

### 20A. Dashboard Framework

**Recommended:** Streamlit (Python)

Reasoning:
- Python-native
- Easy to build with data science tools
- Handles real-time updates with st.rerun
- Plotly integration for interactive charts
- Simple to deploy

### 20B. Dashboard Sections

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

### 20C. API Endpoints

For integration with other systems:

| Endpoint | Method | Returns |
|---|---|---|
| `/api/status` | GET | Current flare status |
| `/api/forecast` | GET | Forecast probabilities for all horizons |
| `/api/catalog` | GET | Master flare catalog |
| `/api/nowcast` | GET | Nowcasting alert details |

### 20D. Real-Time Update Cycle

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

## 21. Ablation Study Plan

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

## 22. Research-Grade Experimentation Plan

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

## 23. Implementation Roadmap

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
