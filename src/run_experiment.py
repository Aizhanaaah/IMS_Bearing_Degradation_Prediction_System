import sys

import pandas as pd

from models import run


EXPERIMENTS = [

    {"EXPERIMENT_NAME": "baseline"},

    {"EXPERIMENT_NAME": "norm_target",
     "TARGET": "rul_norm_fpt"},

    {"EXPERIMENT_NAME": "bearing_weights",
     "BEARING_WEIGHTS": True},

    {"EXPERIMENT_NAME": "norm_target_weights",
     "TARGET": "rul_norm_fpt",
     "BEARING_WEIGHTS": True},
]


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

    for exp in selected:
        settings = dict(exp)
        if quick:
            settings["SVR_TUNE"] = False
            settings["EXPERIMENT_NAME"] += "_quick"
        all_results.append(run(settings))

    res = pd.concat(all_results, ignore_index=True)

    per_bearing = res.pivot_table(
        index=["experiment", "model"], columns="test_bearing", values="mae_hours")
    bearing_cols = list(per_bearing.columns)
    per_bearing["avg"] = per_bearing[bearing_cols].mean(axis=1)
    per_bearing["std"] = per_bearing[bearing_cols].std(axis=1)
    per_bearing["max"] = per_bearing[bearing_cols].max(axis=1)

    print(f"\n{'=' * 60}\nComparison: test MAE in hours (lower is better)\n")
    print(per_bearing.round(1).to_string())

    best = per_bearing["avg"].idxmin()
    print(f"\nBest average: {best[1]} in '{best[0]}' "
          f"({per_bearing.loc[best, 'avg']:.1f} h)")


if __name__ == "__main__":
    main(sys.argv[1:])
