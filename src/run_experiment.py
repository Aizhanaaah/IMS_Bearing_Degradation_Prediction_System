import sys

import pandas as pd

from health_indicator import construct_health_indicator
from models import DEFAULT_SETTINGS, load_data, prepare_features, run
from validation import evaluate_indicator


EXPERIMENTS = [

    {"EXPERIMENT_NAME": "baseline"},

    {"EXPERIMENT_NAME": "norm_target",
     "TARGET": "rul_norm_fpt"},

    {"EXPERIMENT_NAME": "bearing_weights",
     "BEARING_WEIGHTS": True},

    {"EXPERIMENT_NAME": "norm_target_weights",
     "TARGET": "rul_norm_fpt",
     "BEARING_WEIGHTS": True},

    {"EXPERIMENT_NAME": "smooth_5",  "SMOOTH_WINDOW": 5},
    {"EXPERIMENT_NAME": "smooth_10", "SMOOTH_WINDOW": 10},
    {"EXPERIMENT_NAME": "smooth_20", "SMOOTH_WINDOW": 20}, 
]


def hi_behaviour(settings):
    # HI built from the model inputs of each bearing (after smoothing etc.),
    # scored like validation.py does.
    cfg = {**DEFAULT_SETTINGS, **settings}
    df = load_data(cfg)
    features = prepare_features(df, cfg)

    rows = []
    for bearing, group in df.groupby("bearing_id"):
        hi, _ = construct_health_indicator(group[features].reset_index(drop=True), features)
        scores = evaluate_indicator(hi["HI_raw"].to_numpy())
        rows.append({"experiment": cfg["EXPERIMENT_NAME"], "test_bearing": bearing,
                     "monotonicity": scores["monotonicity"],
                     "spearman": scores["spearman_rho"],
                     "roughness": scores["normalized_roughness"]})
    return pd.DataFrame(rows)


def main(argv):

    quick = "--quick" in argv
    names = [a for a in argv if not a.startswith("--")]

    if "--list" in argv:
        for exp in EXPERIMENTS:
            print(exp["EXPERIMENT_NAME"])
        return

    known = {exp["EXPERIMENT_NAME"] for exp in EXPERIMENTS}
    unknown = set(names) - known
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {sorted(unknown)}. "
                         f"Known: {sorted(known)}")
    selected = [exp for exp in EXPERIMENTS
                if not names or exp["EXPERIMENT_NAME"] in names]

    all_results = []
    all_hi = []

    for exp in selected:
        settings = dict(exp)
        if quick:
            settings["SVR_TUNE"] = False
            settings["EXPERIMENT_NAME"] += "_quick"
        all_results.append(run(settings))
        all_hi.append(hi_behaviour(settings))

    res = pd.concat(all_results, ignore_index=True)

    per_bearing = res.pivot_table(
        index=["experiment", "model"], columns="test_bearing", values="mae_hours")
    bearing_cols = list(per_bearing.columns)
    per_bearing["avg"] = per_bearing[bearing_cols].mean(axis=1)
    per_bearing["std"] = per_bearing[bearing_cols].std(axis=1)
    per_bearing["max"] = per_bearing[bearing_cols].max(axis=1)

    print(f"\n{'=' * 60}\nComparison: test MAE in hours (lower is better)\n")
    print(per_bearing.round(1).to_string())

    hi = pd.concat(all_hi).groupby("experiment").mean(numeric_only=True)
    mae = per_bearing["avg"].unstack("model").add_prefix("MAE_")
    print(f"\n{'=' * 60}\nHI behaviour of the model inputs vs. average test MAE (h)\n")
    print(hi.join(mae).round(3).to_string())

    best = per_bearing["avg"].idxmin()
    print(f"\nBest average: {best[1]} in '{best[0]}' "
          f"({per_bearing.loc[best, 'avg']:.1f} h)")


if __name__ == "__main__":
    main(sys.argv[1:])
