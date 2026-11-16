"""
models.py

Linear model + Random Forest + SVR for bearing RUL, with every improvement idea
behind a switch in the EXPERIMENT SETTINGS section. Change ONE setting at a
time, run, and compare the results in experiment_log_v2.csv.

Protocol (fixed, leakage-free):
  Round 1: train T2_B1, T3_B3               -> test T1_B4
  Round 2: train T1_B3, T1_B4, T3_B3        -> test T2_B1
  Round 3: train T1_B3, T1_B4, T2_B1        -> test T3_B3

Everything that learns something (feature selection, duplicate removal,
scaling, the weights, the trees) uses the TRAINING bearings of the round only.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# ==================================================================
# EXPERIMENT SETTINGS  <-- this is your playground
# ==================================================================

EXPERIMENT_NAME = "rerun_after_fixes"   # give every run a short name

DATASET = "zscore"               # "zscore" or "raw"

FEATURE_SET = "physics"           # "physics" (your 9) or "all" (53)

REMOVE_DUPLICATES = True        # drop one of each highly correlated pair
DUPLICATE_THRESHOLD = 0.90

SELECT_BY_RULE = False           # keep only features that trend in EVERY training bearing
MIN_ABS_SPEARMAN = 0.30

SMOOTH_WINDOW = 1               # causal moving average over the last N recordings (1 = off)

ADD_RUNNING_MAX = False          # add "highest value so far" versions of some features
RUNNING_MAX_FEATURES = [
    "rms", "peak_to_peak",
    "BPFI_H1_band_power", "BPFO_H1_band_power", "BSF_H1_band_power",
]

LOG_TRANSFORM = False            # sign(x) * log(1 + |x|)

# Linear model
LINEAR_MODEL = "linear"          # "linear" or "ridge"
RIDGE_ALPHA = 100
CLIP_PREDICTIONS = True          # force Linear/SVR predictions into [0, max training RUL]

# Random Forest
NUMBER_OF_TREES = 300
MIN_SAMPLES_LEAF = 20
MAX_FEATURES = "sqrt"            # "sqrt", "log2", or None (all features)

# SVR (features AND target are standardized inside the model)
SVR_KERNEL = "rbf"               # "rbf" or "linear"
SVR_TUNE = True                  # True = inner leave-one-test-out grid search
SVR_C = 10                       # used when SVR_TUNE = False
SVR_GAMMA = "scale"              # used when SVR_TUNE = False (ignored by "linear")
SVR_EPSILON = 0.1                # in standardized-target units (0.1 = 0.1 std of RUL)
SVR_GRID = {
    "regressor__svr__C": [0.01, 0.1, 1, 10, 100, 1000],
    "regressor__svr__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 1],
    "regressor__svr__epsilon": [0.05, 0.1, 0.2, 0.5, 1.0],
}

# ==================================================================
# Fixed settings (you normally don't touch these)
# ==================================================================

PROCESSED = Path(__file__).resolve().parent.parent / "Data" / "processed"
DATA_FILE = PROCESSED / f"ml_dataset_{DATASET}.csv"
LOG_FILE = PROCESSED / "experiment_log_v2.csv"
PLOT_DIR = PROCESSED / "plots" / "experiments" / EXPERIMENT_NAME

TARGET = "rul_piecewise_hours"

META = [
    "bearing_id", "test_group", "recording", "hours_from_start", "stage",
    "rul_hours", "rul_norm_total", "rul_piecewise_hours", "rul_norm_fpt",
]

PHYSICS_SET = [
    "rms", "kurtosis", "crest_factor", "skewness", "peak_to_peak",
    "spectral_entropy",
    "BPFI_H1_band_power", "BPFO_H1_band_power", "BSF_H1_band_power",
]

ROUNDS = [
    {"train": ["T2_B1", "T3_B3"],          "test": "T1_B4"},
    {"train": ["T1_B3", "T1_B4", "T3_B3"], "test": "T2_B1"},
    {"train": ["T1_B3", "T1_B4", "T2_B1"], "test": "T3_B3"},
]


# ==================================================================
# 1. Load data
# ==================================================================

df = pd.read_csv(DATA_FILE)
df = df[df["stage"] == 1].copy()

all_features = [c for c in df.columns if c not in META]
candidates = list(PHYSICS_SET) if FEATURE_SET == "physics" else list(all_features)


# ==================================================================
# 2. Feature transformations (per bearing, past only, no target used)
# ==================================================================

if SMOOTH_WINDOW > 1:
    # Each value = average of itself and the previous SMOOTH_WINDOW - 1
    # recordings of the SAME bearing. No future values.
    df[candidates] = (
        df.groupby("bearing_id")[candidates]
        .transform(lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1).mean())
    )

if ADD_RUNNING_MAX:
    # Highest value seen so far (since FPT) in the SAME bearing.
    # Can never go down, like real damage. Uses only past values.
    for f in RUNNING_MAX_FEATURES:
        new_name = f"{f}_runmax"
        df[new_name] = df.groupby("bearing_id")[f].cummax()
        candidates.append(new_name)

if LOG_TRANSFORM:
    df[candidates] = np.sign(df[candidates]) * np.log1p(df[candidates].abs())


# ==================================================================
# 3. Feature selection helpers (TRAINING bearings only)
# ==================================================================

def spearman_per_bearing(train, features):
    """Rows = features, columns = training bearings, values = Spearman rho with RUL."""
    table = {}
    for bearing, group in train.groupby("bearing_id"):
        table[bearing] = {
            f: spearmanr(group[f], group[TARGET]).statistic for f in features
        }
    return pd.DataFrame(table).fillna(0.0)


def select_by_rule(train, features):
    """Keep a feature only if it has the same sign and |rho| >= MIN_ABS_SPEARMAN
    in every training bearing. If nothing passes, keep the top 3 by mean |rho|."""
    rho = spearman_per_bearing(train, features)

    same_sign = (np.sign(rho).nunique(axis=1) == 1)
    strong = (rho.abs().min(axis=1) >= MIN_ABS_SPEARMAN)

    kept = rho.index[same_sign & strong].tolist()

    if not kept:
        kept = rho.abs().mean(axis=1).sort_values(ascending=False).head(3).index.tolist()
        print("    rule kept nothing -> fallback to top 3 by mean |rho|")

    return kept


def remove_duplicates(train, features):
    """Drop a feature if it is too correlated with a feature already kept.
    Features with a stronger link to RUL are looked at first."""
    rho = spearman_per_bearing(train, features).abs().mean(axis=1)
    ordered = rho.sort_values(ascending=False).index.tolist()

    corr = train[features].corr(method="spearman").abs()

    kept = []
    for f in ordered:
        if all(corr.loc[f, k] < DUPLICATE_THRESHOLD for k in kept):
            kept.append(f)

    return kept


# ==================================================================
# 4. Models
# ==================================================================

class MeanPredictor:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_linear():
    if LINEAR_MODEL == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    return make_pipeline(StandardScaler(), LinearRegression())


def make_forest():
    # No scaler: trees only ask "is x > threshold?", scaling changes nothing.
    return RandomForestRegressor(
        n_estimators=NUMBER_OF_TREES,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=42,
        n_jobs=-1,
    )


def make_svr():
    # Features scaled: the RBF kernel works on distances.
    # Target scaled: epsilon is then in "std of RUL" units, not hours.
    # Predictions are converted back to hours automatically.
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            SVR(kernel=SVR_KERNEL, C=SVR_C, gamma=SVR_GAMMA, epsilon=SVR_EPSILON),
        ),
        transformer=StandardScaler(),
    )


def fit_svr(X_train, y_train, groups):
    """Fit SVR. If SVR_TUNE, choose C/gamma/epsilon by leave-one-TEST-out
    on the TRAINING bearings only (the test bearing is never seen), then refit
    on all training bearings with the best settings.

    `groups` must be the test run (test_group), not the bearing: T1_B3 and
    T1_B4 come from the same run, so splitting them across inner folds would
    make the inner score optimistic (the same leakage the outer protocol
    avoids)."""
    if not SVR_TUNE:
        return make_svr().fit(X_train, y_train), {
            "C": SVR_C, "gamma": SVR_GAMMA, "epsilon": SVR_EPSILON}

    grid = dict(SVR_GRID)
    if SVR_KERNEL == "linear":
        grid.pop("regressor__svr__gamma")      # gamma has no effect on a linear kernel

    search = GridSearchCV(
        make_svr(), grid,
        cv=LeaveOneGroupOut(),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups)

    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    print(f"    SVR best params: {best}  (inner CV MAE={-search.best_score_:.3f})")
    return search.best_estimator_, best


# Models whose output is unbounded and therefore gets clipped.
# (Mean and RandomForest can only return averages of training targets.)
CLIPPED_MODELS = {"Linear", "SVR"}


def clip_rul(pred, y_train):
    """RUL can't be negative or exceed what was seen in training (target is in hours)."""
    return np.clip(pred, 0.0, float(np.max(y_train)))


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


# ==================================================================
# 5. Run the three rounds
# ==================================================================

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Candidates: {len(candidates)} features\n")

results = []
predictions = []
svr_params = {}        # chosen SVR settings per test bearing (for the log)

for r in ROUNDS:

    train = df[df["bearing_id"].isin(r["train"])]
    test = df[df["bearing_id"] == r["test"]]
    assert r["test"] not in set(train["bearing_id"]), "LEAKAGE!"

    # --- choose features (training bearings only), same for all models ---
    features = list(candidates)

    if SELECT_BY_RULE:
        features = select_by_rule(train, features)

    if REMOVE_DUPLICATES:
        features = remove_duplicates(train, features)

    print(f"Test {r['test']}: {len(features)} features -> {features}")

    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET].values

    # --- fit all models on the same training data ---
    svr_model, svr_params[r["test"]] = fit_svr(X_train, y_train, train["test_group"])
    models = {
        "Mean": MeanPredictor().fit(X_train, y_train),
        "Linear": make_linear().fit(X_train, y_train),
        "RandomForest": make_forest().fit(X_train, y_train),
        "SVR": svr_model,
    }

    round_preds = {}

    for name, model in models.items():

        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)

        if name in CLIPPED_MODELS and CLIP_PREDICTIONS:
            pred_test = clip_rul(pred_test, y_train)
            pred_train = clip_rul(pred_train, y_train)

        round_preds[name] = pred_test

        test_m = metrics(y_test, pred_test)
        train_mae = mean_absolute_error(y_train, pred_train)

        results.append({
            "test_bearing": r["test"], "model": name,
            **test_m, "train_mae": train_mae, "n_features": len(features),
        })

        # Diagnostic: training error vs test error.
        #   low train + high test  -> information exists but doesn't transfer
        #   high train + high test -> features carry little information
        print(f"    {name:13s} test MAE={test_m['mae']:.3f}  R2={test_m['r2']:.3f}"
              f"   | train MAE={train_mae:.3f}")

    # --- what each model leans on ---
    lin = models["Linear"]
    weights = pd.Series(lin[-1].coef_, index=features).sort_values()
    print("    linear weights:", weights.round(3).to_dict())

    imp = pd.Series(models["RandomForest"].feature_importances_, index=features)
    print("    RF importances:", imp.sort_values(ascending=False).round(3).to_dict())

    # SVR: share of training rows that are support vectors.
    # Near 100% -> epsilon too small; very low -> epsilon too large / underfit.
    svr_core = models["SVR"].regressor_[-1]
    print(f"    SVR support vectors: {svr_core.support_.size}/{len(X_train)}"
          f" ({svr_core.support_.size / len(X_train):.0%})")

    # SVR has no feature_importances_ (RBF). Permutation importance = how much
    # test MAE gets worse when one feature is shuffled. Interpretation only.
    perm = permutation_importance(
        models["SVR"], X_test, y_test,
        scoring="neg_mean_absolute_error", n_repeats=10, random_state=42, n_jobs=-1,
    )
    perm_imp = pd.Series(perm.importances_mean, index=features)
    print("    SVR permutation importance (MAE increase, h):",
          perm_imp.sort_values(ascending=False).round(3).to_dict(), "\n")

    predictions.append(pd.DataFrame({
        "test_bearing": r["test"],
        "hours": test["hours_from_start"].values,
        "true": y_test,
        **{name: p for name, p in round_preds.items()},
    }))


# ==================================================================
# 6. Summary + experiment log
# ==================================================================

res = pd.DataFrame(results)
summary = res.groupby("model")[["mae", "rmse", "r2", "train_mae"]].mean()

print("Average over the 3 test bearings:")
print(summary.round(3).to_string())

# One line per run: settings + test MAE of every model on every bearing.
row = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "experiment": EXPERIMENT_NAME,
    "dataset": DATASET, "feature_set": FEATURE_SET,
    "remove_duplicates": REMOVE_DUPLICATES, "select_by_rule": SELECT_BY_RULE,
    "smooth": SMOOTH_WINDOW, "running_max": ADD_RUNNING_MAX, "log": LOG_TRANSFORM,
    "linear_model": LINEAR_MODEL,
    "alpha": RIDGE_ALPHA if LINEAR_MODEL == "ridge" else None,
    "trees": NUMBER_OF_TREES, "min_samples_leaf": MIN_SAMPLES_LEAF,
    "svr_kernel": SVR_KERNEL, "svr_tuned": SVR_TUNE,
}
for bearing, p in svr_params.items():
    for k, v in p.items():
        row[f"svr_{k}_{bearing}"] = v
for _, line in res.iterrows():
    row[f"mae_{line['model']}_{line['test_bearing']}"] = round(line["mae"], 4)
for model_name in summary.index:
    row[f"mae_{model_name}_avg"] = round(summary.loc[model_name, "mae"], 4)

# Read the old log (if any), add this row, write everything back.
# This keeps the columns aligned even if new settings are added later.
new_row = pd.DataFrame([row])
if LOG_FILE.exists():
    new_row = pd.concat([pd.read_csv(LOG_FILE), new_row], ignore_index=True)
new_row.to_csv(LOG_FILE, index=False)
print(f"\nLogged to: {LOG_FILE}")


# ==================================================================
# 7. Plots
# ==================================================================

PLOT_DIR.mkdir(parents=True, exist_ok=True)

for p in predictions:
    bearing = p["test_bearing"].iloc[0]

    plt.figure(figsize=(10, 4))
    plt.plot(p["hours"], p["true"], "k", linewidth=2, label="true")
    for name in models:
        plt.plot(p["hours"], p[name], linewidth=1, label=name)
    plt.title(f"{bearing} | {EXPERIMENT_NAME}")
    plt.xlabel("hours from start")
    plt.ylabel(TARGET)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{bearing}.png", dpi=120)
    plt.close()

print(f"Plots saved in: {PLOT_DIR}")