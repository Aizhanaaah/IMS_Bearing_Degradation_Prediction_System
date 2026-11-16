"""
run_experiment.py

Runs several model experiments in one go and prints a comparison table.

Every experiment starts from models.DEFAULT_SETTINGS and changes only the
settings listed in its dictionary (change as few as possible per experiment,
so you know what caused a difference). Every run is also logged to
Data/processed/experiment_log_v2.csv and gets its own plot folder.

Usage (from src/):
  python run_experiment.py                      -> run all EXPERIMENTS
  python run_experiment.py baseline norm_target -> run only these (by name)
  python run_experiment.py --list               -> show the experiment names
  python run_experiment.py --quick ...          -> no SVR grid search (much faster)

Note: with SVR_TUNE = True (the default) one experiment takes several minutes,
because of the SVR grid search.
"""

import sys

import pandas as pd

from models import run


# ==================================================================
# EXPERIMENTS  <-- add, remove or edit experiments here
# ==================================================================

EXPERIMENTS = [
    # Reference: all defaults (hours target, z-score data, 9 physics features).
    {"EXPERIMENT_NAME": "baseline"},

    # Same target scale for every bearing (1 at FPT -> 0 at failure).
    {"EXPERIMENT_NAME": "norm_target",
     "TARGET": "rul_norm_fpt"},

    # Every training bearing counts the same, whatever its number of rows.
    {"EXPERIMENT_NAME": "bearing_weights",
     "BEARING_WEIGHTS": True},

    # Both together.
    {"EXPERIMENT_NAME": "norm_target_weights",
     "TARGET": "rul_norm_fpt",
     "BEARING_WEIGHTS": True},
]


# ==================================================================
# Run + compare
# ==================================================================

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

    # MAE in hours: comparable across experiments, also when the target
    # is normalized.
    per_bearing = res.pivot_table(
        index=["experiment", "model"], columns="test_bearing", values="mae_hours")
    per_bearing["avg"] = per_bearing.mean(axis=1)

    print(f"\n{'=' * 60}\nComparison: test MAE in hours (lower is better)\n")
    print(per_bearing.round(1).to_string())

    best = per_bearing["avg"].idxmin()
    print(f"\nBest average: {best[1]} in '{best[0]}' "
          f"({per_bearing.loc[best, 'avg']:.1f} h)")


if __name__ == "__main__":
    main(sys.argv[1:])
