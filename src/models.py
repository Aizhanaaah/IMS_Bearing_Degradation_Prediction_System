"""
experiments.py

First-pass prediction experiments (no hyper-parameter tuning yet).

Protocol : leave-one-TEST-out (strict, leakage-free) from build_dataset.make_folds
Rows     : only stage == 1 (after FPT), for both training and testing
Target   : rul_norm_fpt  (1 at FPT -> 0 at end of life)
Models   : Dummy(mean), Linear Regression, Random Forest, SVR
Features : three FIXED feature sets, compared for every model
             all        - every unified feature
             top10      - chosen INSIDE each fold, from the training bearings only
             faultband  - fault-frequency features (BPFI/BPFO/BSF/FTF) only

Everything that learns something from y (feature selection, scaling, the model)
is fitted on the training bearings of the fold only.

Run build_dataset.py first.
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from rul_labels import PROCESSED
from build_dataset import make_folds, META_COLUMNS


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_FILE = PROCESSED / "ml_dataset_zscore.csv"   # or ml_dataset_raw.csv

TARGET = "rul_norm_fpt"

FEATURE_SETS = ["all", "top10", "faultband"]
TOP_N = 10

# Bearings used for TRAINING only (never scored). Their test group is still
# held out in its own fold, because T1_B3 and T1_B4 come from the same run:
# in the "hold out test1" fold T1_B3 can be neither a training nor a test bearing.
TRAIN_ONLY = ["T1_B3"]

# A test bearing with fewer post-FPT rows than this is still reported,
# but it is left out of the average (too few points to say anything).
MIN_TEST_ROWS = 100

PLOT_FEATURE_SET = "all"

RESULTS_FILE = PROCESSED / "experiment_results.csv"
PREDICTIONS_FILE = PROCESSED / "experiment_predictions.csv"
PLOT_DIR = PROCESSED / "plots" / "predictions"

FAULT_PREFIXES = ("BPFI_", "BPFO_", "BSF_", "FTF_")


# --------------------------------------------------
# 2. Models (default settings, tuning comes later)
# --------------------------------------------------

def make_models():

    return {
        "Dummy(mean)": DummyRegressor(strategy="mean"),
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ),
        "SVR": make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=1.0, epsilon=0.05),
        ),
    }


# --------------------------------------------------
# 3. Feature sets
# --------------------------------------------------

def select_top_features(train_df, feature_columns, n):
    """
    Rank features by |Spearman correlation| with the target, computed
    SEPARATELY inside every training bearing and then averaged, so that a
    long bearing does not dominate. Uses training bearings only.
    """

    scores = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for _, group in train_df.groupby("bearing_id"):

            if len(group) < 5:
                continue

            scores.append(
                group[feature_columns].apply(
                    lambda column: abs(
                        spearmanr(column, group[TARGET]).statistic
                    )
                )
            )

    mean_score = pd.concat(scores, axis=1).mean(axis=1).fillna(0.0)

    return mean_score.sort_values(ascending=False).head(n).index.tolist()


def get_feature_list(name, train_df, feature_columns):

    if name == "all":
        return list(feature_columns)

    if name == "top10":
        return select_top_features(train_df, feature_columns, TOP_N)

    if name == "faultband":
        return [c for c in feature_columns if c.startswith(FAULT_PREFIXES)]

    raise ValueError(f"Unknown feature set: {name}")


# --------------------------------------------------
# 4. Metrics
# --------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------
# 5. Main experiment loop
# --------------------------------------------------

def run():

    df = pd.read_csv(DATASET_FILE)

    df = df[df["stage"] == 1].copy().reset_index(drop=True)

    feature_columns = [c for c in df.columns if c not in META_COLUMNS]

    print(f"Rows used (stage == 1): {len(df)}")
    print(f"Unified features: {len(feature_columns)}")
    print(df.groupby("bearing_id").size().to_string())

    folds = make_folds(df)

    result_rows = []
    prediction_rows = []

    for fold in folds:

        test_ids = [b for b in fold["test_ids"] if b not in TRAIN_ONLY]

        if not test_ids:
            print(f"\n{fold['name']}: nothing to score, skipped.")
            continue

        train = df[df["bearing_id"].isin(fold["train_ids"])]
        test = df[df["bearing_id"].isin(test_ids)]

        print(
            f"\n{fold['name']}: train={fold['train_ids']} "
            f"({len(train)} rows) -> test={test_ids} "
            f"({len(test)} rows)"
        )

        for feature_set in FEATURE_SETS:

            features = get_feature_list(feature_set, train, feature_columns)

            if not features:
                print(f"  [{feature_set}] no features available, skipped.")
                continue

            if feature_set == "top10":
                print(f"  [top10 chosen from train only] {features}")

            for model_name, model in make_models().items():

                model.fit(train[features], train[TARGET])

                for bearing_id in test_ids:

                    part = test[test["bearing_id"] == bearing_id]

                    predicted = model.predict(part[features])

                    result_rows.append({
                        "fold": fold["name"],
                        "bearing": bearing_id,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n_test": len(part),
                        "mae": mean_absolute_error(part[TARGET], predicted),
                        "rmse": rmse(part[TARGET], predicted),
                    })

                    prediction_rows.append(pd.DataFrame({
                        "bearing": bearing_id,
                        "feature_set": feature_set,
                        "model": model_name,
                        "hours_from_start": part["hours_from_start"].values,
                        "y_true": part[TARGET].values,
                        "y_pred": predicted,
                    }))

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    results.to_csv(RESULTS_FILE, index=False, float_format="%.5f")
    predictions.to_csv(PREDICTIONS_FILE, index=False, float_format="%.5f")

    return results, predictions


# --------------------------------------------------
# 6. Reporting
# --------------------------------------------------

def report(results):

    pivot = results.pivot_table(
        index=["feature_set", "model"],
        columns="bearing",
        values="mae"
    )

    sizes = (
        results.drop_duplicates("bearing")
        .set_index("bearing")["n_test"]
    )

    reliable = [b for b in pivot.columns if sizes[b] >= MIN_TEST_ROWS]

    pivot["mean_MAE"] = pivot[reliable].mean(axis=1)

    print("\n==================================================")
    print("MAE per test bearing (target = rul_norm_fpt, range 0..1)")
    print("==================================================")
    print("Test rows per bearing:", sizes.to_dict())
    print(
        f"mean_MAE uses only bearings with >= {MIN_TEST_ROWS} test rows: "
        f"{reliable}"
    )
    print()
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nRMSE per test bearing")
    print(
        results.pivot_table(
            index=["feature_set", "model"],
            columns="bearing",
            values="rmse"
        ).to_string(float_format=lambda x: f"{x:.3f}")
    )

    print(f"\nSaved: {RESULTS_FILE}")
    print(f"Saved: {PREDICTIONS_FILE}")


def plot_predictions(predictions):

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    subset = predictions[predictions["feature_set"] == PLOT_FEATURE_SET]

    for bearing, group in subset.groupby("bearing"):

        plt.figure(figsize=(11, 4.5))

        truth = group[group["model"] == "Dummy(mean)"]

        plt.plot(
            truth["hours_from_start"],
            truth["y_true"],
            color="black",
            linewidth=2,
            label="true RUL (normalized)"
        )

        for model_name, part in group.groupby("model"):

            plt.plot(
                part["hours_from_start"],
                part["y_pred"],
                linewidth=1,
                label=model_name
            )

        plt.xlabel("Hours from start")
        plt.ylabel("rul_norm_fpt")
        plt.title(f"{bearing}: prediction vs truth ({PLOT_FEATURE_SET} features)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out = PLOT_DIR / f"predictions_{bearing}.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()

        print(f"Plot saved: {out}")


def main():

    results, predictions = run()
    report(results)
    plot_predictions(predictions)


if __name__ == "__main__":
    main()