# IMS Bearing Degradation Prediction System

Thesis project: predicting the **Remaining Useful Life (RUL)** of rolling-element bearings from vibration data in the **NASA IMS bearing dataset** (run-to-failure experiments from the Center for Intelligent Maintenance Systems, University of Cincinnati).

The pipeline goes from raw vibration recordings to time- and frequency-domain features, health indicators, a First Prediction Time (FPT) detector, RUL labels, and finally regression models (Linear, Random Forest, SVR). The models are evaluated with a leave-one-test-out protocol so that no information leaks from the test bearing into training.

---

## Dataset

The IMS dataset has three run-to-failure tests. Four bearings failed, and these four are used for modelling:

| Test | Channels | Valid recordings | Failed bearing(s) | ID used in code |
|------|----------|------------------|-------------------|-----------------|
| 1st_test | 8 (x and y per bearing) | 2156 | Bearing 3, Bearing 4 | `T1_B3`, `T1_B4` |
| 2nd_test | 4 | 982 | Bearing 1 | `T2_B1` |
| 3rd_test | 4 | 6323 | Bearing 3 | `T3_B3` |

Each recording is a 1-second snapshot sampled at **20 kHz**. Shaft speed is 2000 RPM.

The last recordings of tests 2 and 3 read about 0.002 g on every channel, which means the sensors were already switched off after the failure. They are dropped everywhere (see `VALID_RECORDINGS` in `src/data_loader.py`).

> **The raw data is not in this repository** (it is git-ignored). Download the IMS bearing dataset from the NASA Prognostics Center of Excellence data repository and extract it so the folder layout looks like this:
>
> ```
> Data/
> ├── 1st_test/
> ├── 2nd_test/
> └── 3rd_test/
> ```

---

## Repository structure

```
ThesisWork/
├── src/
│   ├── data_loader.py         # Read raw recordings, channel names, valid recording counts
│   ├── feature_extraction.py  # Time/frequency features + bearing fault frequency band powers
│   ├── feature_analysis.py    # Plots of selected features over the life of each bearing
│   ├── feature_evaluation.py  # Monotonicity / trendability / robustness scores per feature
│   ├── health_indicator.py    # Build health indicators from the top-ranked features (LOESS smoothing)
│   ├── validation.py          # Evaluate the health indicators
│   ├── rul_labels.py          # RUL labels + FPT detection (3-sigma rule, consecutive trigger)
│   ├── build_dataset.py       # One unified ML table for the 4 failed bearings + fold definitions
│   ├── models.py              # Linear, Random Forest and SVR RUL models with experiment switches
│   └── run_experiment.py      # (placeholder)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── literature/
│   └── paper_comparison.xlsx  # Comparison of related work
├── results/                   # figures/ and metrics/
├── tests/                     # (placeholders)
└── Data/                      # Raw + processed data (not tracked)
```

---

## Pipeline

Run the scripts from inside `src/` in this order. Each step writes its outputs to `Data/processed/`.

```bash
cd src
python data_loader.py
python feature_extraction.py
python feature_analysis.py
python feature_evaluation.py
python health_indicator.py
python validation.py
python rul_labels.py
python rul_labels.py --sensitivity   # optional: FPT parameter sensitivity study
python build_dataset.py
python models.py
```

### 1. Feature extraction
For every channel and recording:
- **Time domain:** RMS, peak-to-peak, skewness, kurtosis, crest factor, ...
- **Frequency domain:** dominant frequency, spectral centroid, spectral entropy, ...
- **Fault frequencies:** band power around the theoretical BPFI, BPFO, BSF (and FTF) frequencies and their harmonics, calculated from the bearing geometry (16 rolling elements, 8.4 mm ball diameter, 71.5 mm pitch diameter, 15.17° contact angle).

### 2. Feature evaluation and health indicators
Features are scored on **monotonicity**, **trendability** and **robustness**. The best-scoring features are combined into health indicators, which are then validated (monotonicity, Spearman correlation with time, roughness).

### 3. FPT detection and RUL labels
- `rul_hours = T_end − t`, where `T_end` is the last valid recording.
- The **First Prediction Time** is the point where degradation starts. It is detected on the RMS signal with an expanding-window 3-sigma rule: `cv_t = μ_{t−1} + 3·σ_{t−1}`, triggered by 3 consecutive exceedances (min. 50 recordings of history).
- Models are trained only on the post-FPT part, with the target `rul_piecewise_hours`.

| Bearing | FPT (h) | Life fraction at FPT | Post-FPT rows | RUL at FPT (h) |
|---------|---------|----------------------|---------------|----------------|
| T1_B3 | 750.2 | 0.91 | 313 | 77.4 |
| T1_B4 | 718.4 | 0.87 | 489 | 109.2 |
| T2_B1 | 88.7 | 0.54 | 450 | 74.8 |
| T3_B3 | 557.9 | 0.52 | 3044 | 515.2 |

### 4. Unified dataset
`build_dataset.py` turns the per-test feature tables into one table:
- channel-independent feature names (two-channel bearings are averaged),
- constant and NaN/inf features removed,
- optional causal **per-bearing z-score** against the healthy baseline window (`ml_dataset_zscore.csv`) or raw values (`ml_dataset_raw.csv`).

### 5. Models
`models.py` trains a **mean baseline**, **Linear regression**, **Random Forest** and **SVR** (RBF, tuned with an inner leave-one-test-out grid search).

Evaluation protocol (leave-one-test-out, so bearings from the same test never appear in both train and test):

| Round | Train | Test |
|-------|-------|------|
| 1 | T2_B1, T3_B3 | T1_B4 |
| 2 | T1_B3, T1_B4, T3_B3 | T2_B1 |
| 3 | T1_B3, T1_B4, T2_B1 | T3_B3 |

Everything that learns from data (feature selection, duplicate removal, scaling, hyperparameters) is fitted on the training bearings of the round only.

The **EXPERIMENT SETTINGS** block at the top of `models.py` holds the switches for each idea (dataset type, feature set, duplicate removal, smoothing, running-max features, log transform, model hyperparameters, ...). Change one setting at a time, give the run a new `EXPERIMENT_NAME`, and compare runs in `Data/processed/experiment_log_v2.csv`. Plots go to `Data/processed/plots/experiments/<EXPERIMENT_NAME>/`.

---

## Current results

Latest run (`rerun_after_fixes`: z-score dataset, 9 physics-based features), averaged over the 3 test bearings:

| Model | MAE (h) | RMSE (h) | R² |
|-------|---------|----------|-----|
| Mean baseline | 188.3 | 203.6 | −35.8 |
| Linear | 183.4 | 202.4 | −33.8 |
| **Random Forest** | **102.8** | **124.8** | **−4.4** |
| SVR | 138.8 | 159.3 | −15.2 |

Random Forest is the best model so far, with good results on `T2_B1` (MAE ≈ 14.7 h). Generalising to `T3_B3`, whose post-FPT phase is much longer than any training bearing, is still the main open problem. The negative R² values show how hard cross-test generalisation is with only four run-to-failure bearings.

---

## Requirements

Python 3.12 with:

```bash
pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

---

## Author

[@Aizhanaaah](https://github.com/Aizhanaaah), thesis project.
