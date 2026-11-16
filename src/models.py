"""
models.py

Linear model + Random Forest + SVR for bearing RUL, with every improvement idea
behind a switch in DEFAULT_SETTINGS.

Two ways to use it:
  python models.py            -> one run with DEFAULT_SETTINGS
  python run_experiment.py    -> several runs, each changing a few settings
                                 (see EXPERIMENTS in run_experiment.py)

Every run adds one line to experiment_log_v2.csv.

Protocol (fixed, leakage-free, leave-one-TEST-out from build_dataset.make_folds):
  Round 1: train T2_B1, T3_B3               -> test T1_B3, T1_B4
  Round 2: train T1_B3, T1_B4, T3_B3        -> test T2_B1
  Round 3: train T1_B3, T1_B4, T2_B1        -> test T3_B3
Metrics are computed per test bearing and averaged over the 4 bearings.

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

from build_dataset import make_folds

# ==================================================================
# DEFAULT SETTINGS  <-- every experiment starts from these
# ==================================================================

DEFAULT_SETTINGS = {
    "EXPERIMENT_NAME": "default",     # give every run a short name

    "DATASET": "zscore",              # "zscore" or "raw"
    "FEATURE_SET": "physics",         # "physics" (the 9 below) or "all" (53)

    # Target the models learn:
    #   "rul_piecewise_hours" -> RUL in hours (flat before FPT, linear after)
    #   "rul_norm_fpt"        -> RUL / RUL at FPT, goes 1 -> 0 for EVERY bearing,
    #                            so bearings with very different lifetimes
    #                            (T3_B3: 515 h, the others: 75-109 h) share one scale.
    "TARGET": "rul_piecewise_hours",

    # True -> every training bearing gets the same total weight, whatever its
    # number of rows (otherwise T3_B3, ~80% of the rows, dominates training).
    "BEARING_WEIGHTS": False,

    "REMOVE_DUPLICATES": True,        # drop one of each highly correlated pair
    "DUPLICATE_THRESHOLD": 0.90,

    "SELECT_BY_RULE": False,          # keep only features that trend in EVERY training bearing
    "MIN_ABS_SPEARMAN": 0.30,

    "SMOOTH_WINDOW": 1,               # causal moving average over the last N recordings (1 = off)

    "ADD_RUNNING_MAX": False,         # add "highest value so far" versions of some features
    "RUNNING_MAX_FEATURES": [
        "rms", "peak_to_peak",
        "BPFI_H1_band_power", "BPFO_H1_band_power", "BSF_H1_band_power",
    ],

    "LOG_TRANSFORM": False,           # sign(x) * log(1 + |x|)

    # Linear model
    "LINEAR_MODEL": "linear",         # "linear" or "ridge"
    "RIDGE_ALPHA": 100,
    "CLIP_PREDICTIONS": True,         # force Linear/SVR predictions into [0, max training target]

    # Random Forest
    "NUMBER_OF_TREES": 300,
    "MIN_SAMPLES_LEAF": 20,
    "MAX_FEATURES": "sqrt",           # "sqrt", "log2", or None (all features)

    # SVR (features AND target are standardized inside the model)
    "SVR_KERNEL": "rbf",              # "rbf" or "linear"
    "SVR_TUNE": True,                 # True = inner leave-one-test-out grid search (slow)
    "SVR_C": 10,                      # used when SVR_TUNE = False
    "SVR_GAMMA": "scale",             # used when SVR_TUNE = False (ignored by "linear")
    "SVR_EPSILON": 0.1,               # in standardized-target units (0.1 = 0.1 std of target)
    "SVR_GRID": {
        "regressor__svr__C": [0.01, 0.1, 1, 10, 100, 1000],
        "regressor__svr__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 1],
        "regressor__svr__epsilon": [0.05, 0.1, 0.2, 0.5, 1.0],
    },
}

# ==================================================================
# Fixed settings (you normally don't touch these)
# ==================================================================

PROCESSED = Path(__file__).resolve().parent.parent / "Data" / "processed"
LOG_FILE = PROCESSED / "experiment_log_v2.csv"

TARGETS = ("rul_piecewise_hours", "rul_norm_fpt")

META = [
    "bearing_id", "test_group", "recording", "hours_from_start", "stage",
    "rul_hours", "rul_norm_total", "rul_piecewise_hours", "rul_norm_fpt",
]

PHYSICS_SET = [
    "rms", "kurtosis", "crest_factor", "skewness", "peak_to_peak",
    "spectral_entropy",
    "BPFI_H1_band_power", "BPFO_H1_band_power", "BSF_H1_band_power",
]

# Models whose output is unbounded and therefore gets clipped.
# (Mean and RandomForest can only return averages of training targets.)
CLIPPED_MODELS = {"Linear", "SVR"}


# ==================================================================
# 1. Data + feature transformations (per bearing, past only, no target used)
# ==================================================================

def load_data(cfg):
    df = pd.read_csv(PROCESSED / f"ml_dataset_{cfg['DATASET']}.csv")
    df = df[df["stage"] == 1].copy()

    # RUL at the FPT of each bearing (first post-FPT row). Only used to
    # convert normalized predictions back to hours for reporting.
    df["rul_at_fpt"] = df.groupby("bearing_id")["rul_hours"].transform("first")
    return df


def prepare_features(df, cfg):
    """Returns the candidate feature list (and adds columns to df in place)."""
    all_features = [c for c in df.columns if c not in META + ["rul_at_fpt"]]
    candidates = (list(PHYSICS_SET) if cfg["FEATURE_SET"] == "physics"
                  else list(all_features))

    if cfg["SMOOTH_WINDOW"] > 1:
        # Each value = average of itself and the previous SMOOTH_WINDOW - 1
        # recordings of the SAME bearing. No future values.
        w = cfg["SMOOTH_WINDOW"]
        df[candidates] = (
            df.groupby("bearing_id")[candidates]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )

    if cfg["ADD_RUNNING_MAX"]:
        # Highest value seen so far (since FPT) in the SAME bearing.
        # Can never go down, like real damage. Uses only past values.
        for f in cfg["RUNNING_MAX_FEATURES"]:
            new_name = f"{f}_runmax"
            df[new_name] = df.groupby("bearing_id")[f].cummax()
            candidates.append(new_name)

    if cfg["LOG_TRANSFORM"]:
        df[candidates] = np.sign(df[candidates]) * np.log1p(df[candidates].abs())

    return candidates


# ==================================================================
# 2. Feature selection helpers (TRAINING bearings only)
# ==================================================================

def spearman_per_bearing(train, features, target):
    """Rows = features, columns = training bearings, values = Spearman rho with RUL."""
    table = {}
    for bearing, group in train.groupby("bearing_id"):
        table[bearing] = {
            f: spearmanr(group[f], group[target]).statistic for f in features
        }
    return pd.DataFrame(table).fillna(0.0)


def select_by_rule(train, features, cfg):
    """Keep a feature only if it has the same sign and |rho| >= MIN_ABS_SPEARMAN
    in every training bearing. If nothing passes, keep the top 3 by mean |rho|."""
    rho = spearman_per_bearing(train, features, cfg["TARGET"])

    same_sign = (np.sign(rho).nunique(axis=1) == 1)
    strong = (rho.abs().min(axis=1) >= cfg["MIN_ABS_SPEARMAN"])

    kept = rho.index[same_sign & strong].tolist()

    if not kept:
        kept = rho.abs().mean(axis=1).sort_values(ascending=False).head(3).index.tolist()
        print("    rule kept nothing -> fallback to top 3 by mean |rho|")

    return kept


def remove_duplicates(train, features, cfg):
    """Drop a feature if it is too correlated with a feature already kept.
    Features with a stronger link to RUL are looked at first."""
    rho = spearman_per_bearing(train, features, cfg["TARGET"]).abs().mean(axis=1)
    ordered = rho.sort_values(ascending=False).index.tolist()

    corr = train[features].corr(method="spearman").abs()

    kept = []
    for f in ordered:
        if all(corr.loc[f, k] < cfg["DUPLICATE_THRESHOLD"] for k in kept):
            kept.append(f)

    return kept


def bearing_weights(train):
    """Weight = 1 / (rows of that bearing), rescaled to mean 1, so that every
    training bearing contributes the same total weight."""
    counts = train["bearing_id"].map(train["bearing_id"].value_counts())
    w = 1.0 / counts.to_numpy(dtype=float)
    return w / w.mean()


# ==================================================================
# 3. Models
# ==================================================================

class MeanPredictor:
    def fit(self, X, y, sample_weight=None):
        self.mean_ = float(np.average(y, weights=sample_weight))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_linear(cfg):
    if cfg["LINEAR_MODEL"] == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=cfg["RIDGE_ALPHA"]))
    return make_pipeline(StandardScaler(), LinearRegression())


def fit_linear(X, y, w, cfg):
    model = make_linear(cfg)
    last_step = model.steps[-1][0]           # "linearregression" or "ridge"
    return model.fit(X, y, **({f"{last_step}__sample_weight": w} if w is not None else {}))


def make_forest(cfg):
    # No scaler: trees only ask "is x > threshold?", scaling changes nothing.
    return RandomForestRegressor(
        n_estimators=cfg["NUMBER_OF_TREES"],
        min_samples_leaf=cfg["MIN_SAMPLES_LEAF"],
        max_features=cfg["MAX_FEATURES"],
        random_state=42,
        n_jobs=-1,
    )


def make_svr(cfg):
    # Features scaled: the RBF kernel works on distances.
    # Target scaled: epsilon is then in "std of target" units.
    # Predictions are converted back to target units automatically.
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            SVR(kernel=cfg["SVR_KERNEL"], C=cfg["SVR_C"],
                gamma=cfg["SVR_GAMMA"], epsilon=cfg["SVR_EPSILON"]),
        ),
        transformer=StandardScaler(),
    )


def fit_svr(X_train, y_train, groups, w, cfg):
    """Fit SVR. If SVR_TUNE, choose C/gamma/epsilon by leave-one-TEST-out
    on the TRAINING bearings only (the test bearing is never seen), then refit
    on all training bearings with the best settings.

    `groups` must be the test run (test_group), not the bearing: T1_B3 and
    T1_B4 come from the same run, so splitting them across inner folds would
    make the inner score optimistic (the same leakage the outer protocol
    avoids)."""
    fit_params = {"svr__sample_weight": w} if w is not None else {}

    if not cfg["SVR_TUNE"]:
        return make_svr(cfg).fit(X_train, y_train, **fit_params), {
            "C": cfg["SVR_C"], "gamma": cfg["SVR_GAMMA"], "epsilon": cfg["SVR_EPSILON"]}

    grid = dict(cfg["SVR_GRID"])
    if cfg["SVR_KERNEL"] == "linear":
        grid.pop("regressor__svr__gamma")      # gamma has no effect on a linear kernel

    search = GridSearchCV(
        make_svr(cfg), grid,
        cv=LeaveOneGroupOut(),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train, groups=groups, **fit_params)

    best = {k.split("__")[-1]: v for k, v in search.best_params_.items()}
    print(f"    SVR best params: {best}  (inner CV MAE={-search.best_score_:.3f})")
    return search.best_estimator_, best


def clip_target(pred, y_train):
    """RUL can't be negative or exceed what was seen in training."""
    return np.clip(pred, 0.0, float(np.max(y_train)))


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


# ==================================================================
# 4. One complete run
# ==================================================================

def run(settings=None):
    """Run all rounds with DEFAULT_SETTINGS updated by `settings`.
    Logs one line to experiment_log_v2.csv, saves plots, and returns the
    per-bearing results as a DataFrame."""

    settings = settings or {}
    unknown = set(settings) - set(DEFAULT_SETTINGS)
    if unknown:
        raise KeyError(f"Unknown setting(s): {sorted(unknown)}")

    cfg = {**DEFAULT_SETTINGS, **settings}
    if cfg["TARGET"] not in TARGETS:
        raise ValueError(f"TARGET must be one of {TARGETS}")

    target = cfg["TARGET"]
    normalized = target == "rul_norm_fpt"
    name = cfg["EXPERIMENT_NAME"]

    df = load_data(cfg)
    candidates = prepare_features(df, cfg)

    # Same folds as build_dataset.py: all bearings of one test run are held out
    # together (T1_B3 and T1_B4 share a run, so they are always tested together).
    rounds = [
        {"train": f["train_ids"], "test": f["test_ids"]}
        for f in make_folds(df)
    ]

    print(f"\n{'=' * 60}\nExperiment: {name}")
    changed = {k: v for k, v in settings.items() if k != "EXPERIMENT_NAME"}
    print(f"Changed settings: {changed or 'none (defaults)'}")
    print(f"Target: {target}   Candidates: {len(candidates)} features\n")

    results = []
    predictions = []
    svr_params = {}        # chosen SVR settings per round (for the log)

    for r in rounds:

        train = df[df["bearing_id"].isin(r["train"])]
        test = df[df["bearing_id"].isin(r["test"])]
        assert not set(r["test"]) & set(train["bearing_id"]), "LEAKAGE!"
        test_label = "+".join(r["test"])

        # --- choose features (training bearings only), same for all models ---
        features = list(candidates)

        if cfg["SELECT_BY_RULE"]:
            features = select_by_rule(train, features, cfg)

        if cfg["REMOVE_DUPLICATES"]:
            features = remove_duplicates(train, features, cfg)

        print(f"Test {test_label}: {len(features)} features -> {features}")

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target].values
        w = bearing_weights(train) if cfg["BEARING_WEIGHTS"] else None

        # --- fit all models on the same training data ---
        svr_model, svr_params[test_label] = fit_svr(
            X_train, y_train, train["test_group"], w, cfg)
        models = {
            "Mean": MeanPredictor().fit(X_train, y_train, sample_weight=w),
            "Linear": fit_linear(X_train, y_train, w, cfg),
            "RandomForest": make_forest(cfg).fit(X_train, y_train, sample_weight=w),
            "SVR": svr_model,
        }

        round_preds = {}

        for model_name, model in models.items():

            pred_test = model.predict(X_test)
            pred_train = model.predict(X_train)

            if model_name in CLIPPED_MODELS and cfg["CLIP_PREDICTIONS"]:
                pred_test = clip_target(pred_test, y_train)
                pred_train = clip_target(pred_train, y_train)

            round_preds[model_name] = pred_test

            train_mae = mean_absolute_error(y_train, pred_train)

            # Predictions in hours. For the normalized target this multiplies
            # by the TRUE RUL at FPT of the test bearing, which is only known
            # after the failure -> use mae_hours to compare experiments,
            # not as a deployable result.
            if normalized:
                pred_hours = pred_test * test["rul_at_fpt"].values
            else:
                pred_hours = pred_test

            # Metrics per test bearing (round 1 has two test bearings).
            for bearing in r["test"]:
                mask = (test["bearing_id"] == bearing).values
                test_m = metrics(y_test[mask], pred_test[mask])
                mae_hours = mean_absolute_error(
                    test["rul_hours"].values[mask], pred_hours[mask])

                results.append({
                    "test_bearing": bearing, "model": model_name,
                    **test_m, "mae_hours": mae_hours,
                    "train_mae": train_mae, "n_features": len(features),
                })

                # Diagnostic: training error vs test error.
                #   low train + high test  -> information exists but doesn't transfer
                #   high train + high test -> features carry little information
                hours_note = f"  (MAE {mae_hours:.1f} h)" if normalized else ""
                print(f"    {model_name:13s} {bearing}  test MAE={test_m['mae']:.3f}"
                      f"  R2={test_m['r2']:.3f}   | train MAE={train_mae:.3f}{hours_note}")

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
        print("    SVR permutation importance (MAE increase):",
              perm_imp.sort_values(ascending=False).round(3).to_dict(), "\n")

        predictions.append(pd.DataFrame({
            "test_bearing": test["bearing_id"].values,
            "hours": test["hours_from_start"].values,
            "true": y_test,
            **round_preds,
        }))

    res = pd.DataFrame(results)
    res.insert(0, "experiment", name)
    summary = res.groupby("model")[["mae", "rmse", "r2", "mae_hours", "train_mae"]].mean()

    print(f"Average over the {res['test_bearing'].nunique()} test bearings:")
    print(summary.round(3).to_string())

    write_log(cfg, res, summary, svr_params)
    save_plots(cfg, predictions, list(models))

    return res


# ==================================================================
# 5. Experiment log + plots
# ==================================================================

def write_log(cfg, res, summary, svr_params):
    """One line per run: settings + test MAE of every model on every bearing.
    mae_*   -> in target units (hours, or 0-1 for rul_norm_fpt)
    maeh_*  -> in hours (comparable across targets)"""
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": cfg["EXPERIMENT_NAME"],
        "dataset": cfg["DATASET"], "feature_set": cfg["FEATURE_SET"],
        "target": cfg["TARGET"], "bearing_weights": cfg["BEARING_WEIGHTS"],
        "remove_duplicates": cfg["REMOVE_DUPLICATES"],
        "select_by_rule": cfg["SELECT_BY_RULE"],
        "smooth": cfg["SMOOTH_WINDOW"], "running_max": cfg["ADD_RUNNING_MAX"],
        "log": cfg["LOG_TRANSFORM"],
        "linear_model": cfg["LINEAR_MODEL"],
        "alpha": cfg["RIDGE_ALPHA"] if cfg["LINEAR_MODEL"] == "ridge" else None,
        "trees": cfg["NUMBER_OF_TREES"], "min_samples_leaf": cfg["MIN_SAMPLES_LEAF"],
        "svr_kernel": cfg["SVR_KERNEL"], "svr_tuned": cfg["SVR_TUNE"],
    }
    for round_label, p in svr_params.items():
        for k, v in p.items():
            row[f"svr_{k}_{round_label}"] = v
    for _, line in res.iterrows():
        row[f"mae_{line['model']}_{line['test_bearing']}"] = round(line["mae"], 4)
        row[f"maeh_{line['model']}_{line['test_bearing']}"] = round(line["mae_hours"], 4)
    for model_name in summary.index:
        row[f"mae_{model_name}_avg"] = round(summary.loc[model_name, "mae"], 4)
        row[f"maeh_{model_name}_avg"] = round(summary.loc[model_name, "mae_hours"], 4)

    # Read the old log (if any), add this row, write everything back.
    # This keeps the columns aligned even if new settings are added later.
    new_row = pd.DataFrame([row])
    if LOG_FILE.exists():
        new_row = pd.concat([pd.read_csv(LOG_FILE), new_row], ignore_index=True)
    new_row.to_csv(LOG_FILE, index=False)
    print(f"\nLogged to: {LOG_FILE}")


def save_plots(cfg, predictions, model_names):
    plot_dir = PROCESSED / "plots" / "experiments" / cfg["EXPERIMENT_NAME"]
    plot_dir.mkdir(parents=True, exist_ok=True)

    for bearing, p in pd.concat(predictions).groupby("test_bearing"):
        plt.figure(figsize=(10, 4))
        plt.plot(p["hours"], p["true"], "k", linewidth=2, label="true")
        for model_name in model_names:
            plt.plot(p["hours"], p[model_name], linewidth=1, label=model_name)
        plt.title(f"{bearing} | {cfg['EXPERIMENT_NAME']}")
        plt.xlabel("hours from start")
        plt.ylabel(cfg["TARGET"])
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{bearing}.png", dpi=120)
        plt.close()

    print(f"Plots saved in: {plot_dir}")


if __name__ == "__main__":
    run()
