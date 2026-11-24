from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
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


DEFAULT_SETTINGS = {
    "EXPERIMENT_NAME": "default",

    "DATASET": "zscore",
    "FEATURE_SET": "physics",

    "TARGET": "rul_piecewise_hours",

    "BEARING_WEIGHTS": False,

    "REMOVE_DUPLICATES": True,
    "DUPLICATE_THRESHOLD": 0.90,
    "SELECT_BY_RULE": False,
    "MIN_ABS_SPEARMAN": 0.30,
    "SMOOTH_WINDOW": 1,
    "ADD_RUNNING_MAX": False,
    "RUNNING_MAX_FEATURES": [
        "rms", "peak_to_peak",
        "BPFI_H1_band_power", "BPFO_H1_band_power", "BSF_H1_band_power",
    ],
    "LOG_TRANSFORM": False,

    "LINEAR_MODEL": "linear",
    "RIDGE_ALPHA": 100,
    "CLIP_PREDICTIONS": True,

    "NUMBER_OF_TREES": 300,
    "MIN_SAMPLES_LEAF": 20,
    "MAX_FEATURES": "sqrt",

    "SVR_KERNEL": "rbf",
    "SVR_TUNE": True,
    "SVR_C": 10,
    "SVR_GAMMA": "scale",
    "SVR_EPSILON": 0.1,
    "SVR_GRID": {
        "regressor__svr__C": [0.01, 0.1, 1, 10, 100, 1000],
        "regressor__svr__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1, 1],
        "regressor__svr__epsilon": [0.05, 0.1, 0.2, 0.5, 1.0],
    },
}


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


CLIPPED_MODELS = {"Linear", "SVR"}


def load_data(cfg):
    df = pd.read_csv(PROCESSED / f"ml_dataset_{cfg['DATASET']}.csv")
    df = df[df["stage"] == 1].copy()

    df["rul_at_fpt"] = df.groupby("bearing_id")["rul_hours"].transform("first")
    return df


def prepare_features(df, cfg):

    all_features = [c for c in df.columns if c not in META + ["rul_at_fpt"]]
    candidates = (list(PHYSICS_SET) if cfg["FEATURE_SET"] == "physics"
                  else list(all_features))

    if cfg["SMOOTH_WINDOW"] > 1:
        w = cfg["SMOOTH_WINDOW"]
        df[candidates] = (
            df.groupby("bearing_id")[candidates]
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )

    if cfg["ADD_RUNNING_MAX"]:

        for f in cfg["RUNNING_MAX_FEATURES"]:
            new_name = f"{f}_runmax"
            df[new_name] = df.groupby("bearing_id")[f].cummax()
            candidates.append(new_name)
    if cfg["LOG_TRANSFORM"]:
        df[candidates] = np.sign(df[candidates]) * np.log1p(df[candidates].abs())

    return candidates


def spearman_per_bearing(X, y, groups, features):

    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    table = {}
    for bearing in np.unique(groups):
        mask = groups == bearing
        table[bearing] = {
            f: spearmanr(X[f].to_numpy()[mask], y[mask]).statistic for f in features
        }
    return pd.DataFrame(table).fillna(0.0)


def select_by_rule(X, y, groups, features, cfg, verbose=True):

    rho = spearman_per_bearing(X, y, groups, features)

    same_sign = (np.sign(rho).nunique(axis=1) == 1)
    strong = (rho.abs().min(axis=1) >= cfg["MIN_ABS_SPEARMAN"])
    kept = rho.index[same_sign & strong].tolist()

    if not kept:
        kept = rho.abs().mean(axis=1).sort_values(ascending=False).head(3).index.tolist()
        if verbose:
            print("    rule kept nothing -> fallback to top 3 by mean |rho|")

    return kept


def remove_duplicates(X, y, groups, features, cfg):

    rho = spearman_per_bearing(X, y, groups, features).abs().mean(axis=1)
    ordered = rho.sort_values(ascending=False).index.tolist()

    corr = X[features].corr(method="spearman").abs()

    kept = []
    for f in ordered:
        if all(corr.loc[f, k] < cfg["DUPLICATE_THRESHOLD"] for k in kept):
            kept.append(f)

    return kept


def select_features(X, y, groups, cfg, verbose=True):

    features = list(X.columns)

    if cfg["SELECT_BY_RULE"]:
        features = select_by_rule(X, y, groups, features, cfg, verbose)

    if cfg["REMOVE_DUPLICATES"]:
        features = remove_duplicates(X, y, groups, features, cfg)

    return features


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cfg=None):
        self.cfg = cfg

    def fit(self, X, y):
        self.features_ = select_features(X, y, X.index.to_numpy(), self.cfg,
                                         verbose=False)
        return self

    def transform(self, X):
        return X[self.features_]


def bearing_weights(train):

    counts = train["bearing_id"].map(train["bearing_id"].value_counts())
    w = 1.0 / counts.to_numpy(dtype=float)
    return w / w.mean()


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
    last_step = model.steps[-1][0]
    return model.fit(X, y, **({f"{last_step}__sample_weight": w} if w is not None else {}))

def make_forest(cfg):

    return RandomForestRegressor(
        n_estimators=cfg["NUMBER_OF_TREES"],
        min_samples_leaf=cfg["MIN_SAMPLES_LEAF"],
        max_features=cfg["MAX_FEATURES"],
        random_state=42,
        n_jobs=-1,
    )


def make_svr(cfg):

    return TransformedTargetRegressor(
        regressor=make_pipeline(
            FeatureSelector(cfg),
            StandardScaler(),
            SVR(kernel=cfg["SVR_KERNEL"], C=cfg["SVR_C"],
                gamma=cfg["SVR_GAMMA"], epsilon=cfg["SVR_EPSILON"]),
        ),
        transformer=StandardScaler(),
    )


def fit_svr(X_train, y_train, groups, w, cfg):

    fit_params = {"svr__sample_weight": w} if w is not None else {}

    if not cfg["SVR_TUNE"]:
        return make_svr(cfg).fit(X_train, y_train, **fit_params), {
            "C": cfg["SVR_C"], "gamma": cfg["SVR_GAMMA"], "epsilon": cfg["SVR_EPSILON"]}

    grid = dict(cfg["SVR_GRID"])
    if cfg["SVR_KERNEL"] == "linear":
        grid.pop("regressor__svr__gamma")

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

    return np.clip(pred, 0.0, float(np.max(y_train)))


def metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def run(settings=None):

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
    svr_params = {}

    for r in rounds:

        train = df[df["bearing_id"].isin(r["train"])]
        test = df[df["bearing_id"].isin(r["test"])]
        assert not set(r["test"]) & set(train["bearing_id"]), "LEAKAGE!"
        test_label = "+".join(r["test"])

        features = select_features(
            train[candidates], train[target], train["bearing_id"], cfg)

        print(f"Test {test_label}: {len(features)} features -> {features}")

        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target].values
        w = bearing_weights(train) if cfg["BEARING_WEIGHTS"] else None

        X_train_svr = train[candidates].set_index(train["bearing_id"])
        X_test_svr = test[candidates].set_index(test["bearing_id"])
        svr_model, svr_params[test_label] = fit_svr(
            X_train_svr, y_train, train["test_group"], w, cfg)
        svr_features = svr_model.regressor_[0].features_
        assert svr_features == features, "SVR final features differ from the other models"
        models = {
            "Mean": MeanPredictor().fit(X_train, y_train, sample_weight=w),
            "Linear": fit_linear(X_train, y_train, w, cfg),
            "RandomForest": make_forest(cfg).fit(X_train, y_train, sample_weight=w),
            "SVR": svr_model,
        }

        round_preds = {}

        for model_name, model in models.items():

            if model_name == "SVR":
                pred_test = model.predict(X_test_svr)
                pred_train = model.predict(X_train_svr)
            else:
                pred_test = model.predict(X_test)
                pred_train = model.predict(X_train)

            if model_name in CLIPPED_MODELS and cfg["CLIP_PREDICTIONS"]:
                pred_test = clip_target(pred_test, y_train)
                pred_train = clip_target(pred_train, y_train)

            round_preds[model_name] = pred_test

            train_mae = mean_absolute_error(y_train, pred_train)

            if normalized:
                pred_hours = pred_test * test["rul_at_fpt"].values
            else:
                pred_hours = pred_test

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

                hours_note = f"  (MAE {mae_hours:.1f} h)" if normalized else ""
                print(f"    {model_name:13s} {bearing}  test MAE={test_m['mae']:.3f}"
                      f"  R2={test_m['r2']:.3f}   | train MAE={train_mae:.3f}{hours_note}")

        lin = models["Linear"]
        weights = pd.Series(lin[-1].coef_, index=features).sort_values()
        print("    linear weights:", weights.round(3).to_dict())

        imp = pd.Series(models["RandomForest"].feature_importances_, index=features)
        print("    RF importances:", imp.sort_values(ascending=False).round(3).to_dict())

        svr_core = models["SVR"].regressor_[-1]
        print(f"    SVR support vectors: {svr_core.support_.size}/{len(X_train)}"
              f" ({svr_core.support_.size / len(X_train):.0%})")

        perm = permutation_importance(
            models["SVR"], X_test_svr, y_test,
            scoring="neg_mean_absolute_error", n_repeats=10, random_state=42, n_jobs=-1,
        )
        perm_imp = pd.Series(perm.importances_mean, index=candidates)[features]
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


def write_log(cfg, res, summary, svr_params):

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
