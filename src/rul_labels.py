import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_features


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROCESSED = Path(__file__).resolve().parent.parent / "Data" / "processed"

# The number of recordings used per test (and why the last sensor-off
# recordings of tests 2 and 3 are dropped) is defined once in
# data_loader.VALID_RECORDINGS, so that every pipeline step uses the same data.

# Bearings with documented failure -> channels used for the RMS signal.
TARGETS = {
    "1st_test": {3: ["Bx_3", "By_3"], 4: ["Bx_4", "By_4"]},
    "2nd_test": {1: ["B1"]},
    "3rd_test": {3: ["B3"]},
}

# --------------------------------------------------------------------------
# FPT detector: 3-sigma rule with an expanding window and a consecutive
# trigger, following the description in the literature, e.g. the FPT section
# of arXiv 2501.07191 (which builds on the consecutive-trigger idea of
# Lei et al.):
#
#     cv_t = mu_{t-1} + 3 * sigma_{t-1}
#
# where mu and sigma come from ALL previous RMS values. If `consecutive`
# RMS values in a row exceed cv_t, time t is the FPT.
#
# Which parameter comes from where:
#   k = 3              literature (3-sigma rule)
#   consecutive = 3    literature (three consecutive values in 2501.07191;
#                      the Robust-MBDL paper, arXiv 2309.06157, uses two)
#   two_sided = False  AUTHOR'S CHOICE. The papers write mu +/- 3*sigma, but
#                      RMS grows with damage, so only the upper bound is used.
#   min_history = 50   AUTHOR'S CHOICE. The expanding statistics need some
#                      history before they mean anything (50 recordings
#                      ~ 8 hours at 10-minute sampling).
#   reset_gap_hours=24 AUTHOR'S CHOICE (data handling, not from a paper).
#                      Test 1 has a ~148-hour recording gap after which the
#                      RMS level is different. The statistics are restarted
#                      after any gap of at least this length, so that two
#                      monitoring sessions are not mixed. Set to None to
#                      switch it off.
# --------------------------------------------------------------------------
FPT_CONFIG = {
    "k": 3.0,
    "consecutive": 3,
    "min_history": 50,
    "reset_gap_hours": 24.0,
    "two_sided": False,
}

TIME_FORMAT = "%Y.%m.%d.%H.%M.%S"


# --------------------------------------------------
# 2. Base RUL labels (no FPT needed)
# --------------------------------------------------

def load_dataset(dataset):
    """Load the valid recordings' features and build the time axis."""

    f = load_features(dataset)

    t = pd.to_datetime(
        f["recording"].astype(str),
        format=TIME_FORMAT
    )

    if not t.is_monotonic_increasing:
        raise ValueError(f"{dataset}: recordings are not time-ordered.")

    hours = (t - t.iloc[0]).dt.total_seconds() / 3600.0

    return f, t, hours


def make_base_labels(f, t, hours, dataset):
    """RUL = T_end - t, where T_end is the last recording."""

    t_end = hours.iloc[-1]

    labels = pd.DataFrame({
        "recording": f["recording"].values,
        "hours_from_start": hours.values,
        "rul_hours": (t_end - hours).values,
        "rul_norm_total": ((t_end - hours) / t_end).values,
    })

    dt_minutes = hours.diff().dropna() * 60.0
    biggest_gap_pos = int(dt_minutes.idxmax())

    print(
        f"{dataset}: n={len(labels)}, T_end={t_end:.1f} h, "
        f"step median={dt_minutes.median():.1f} min, "
        f"max gap={dt_minutes.max():.1f} min "
        f"(between index {biggest_gap_pos - 1} and {biggest_gap_pos}: "
        f"{t.iloc[biggest_gap_pos - 1]} -> {t.iloc[biggest_gap_pos]})"
    )

    return labels


# --------------------------------------------------
# 3. FPT detection (expanding-window 3-sigma, consecutive trigger)
# --------------------------------------------------

def detect_fpt(
    signal,
    hours=None,
    k=3.0,
    consecutive=3,
    min_history=50,
    reset_gap_hours=24.0,
    two_sided=False
):
    """
    For every time t the threshold is built from all PREVIOUS values
    (inside the current monitoring session):

        cv_t = mu_{t-1} + k * sigma_{t-1}

    The first t for which `consecutive` values in a row (t, t+1, ...) all
    exceed cv_t is the FPT. The detector therefore knows about it
    `consecutive - 1` recordings later (20 minutes for consecutive = 3).

    A new monitoring session (fresh statistics) starts after every recording
    gap of at least `reset_gap_hours`.
    """

    x = np.asarray(signal, dtype=float)
    n = len(x)

    # Start index of the monitoring session each recording belongs to.
    seg_start = np.zeros(n, dtype=int)

    if hours is not None and reset_gap_hours is not None:

        h = np.asarray(hours, dtype=float)
        start = 0

        for i in range(1, n):

            if h[i] - h[i - 1] >= reset_gap_hours:
                start = i

            seg_start[i] = start

    # Cumulative sums give expanding mean/std in O(n).
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])

    mu_t = np.full(n, np.nan)
    sd_t = np.full(n, np.nan)

    for i in range(n):

        s = seg_start[i]
        m = i - s

        if m < min_history:
            continue

        mean = (c1[i] - c1[s]) / m
        var = max((c2[i] - c2[s]) / m - mean ** 2, 0.0)

        mu_t[i] = mean
        sd_t[i] = np.sqrt(var)

    upper_t = mu_t + k * sd_t
    lower_t = mu_t - k * sd_t

    fpt = None

    for t in range(n - consecutive + 1):

        if np.isnan(upper_t[t]):
            continue

        # All points of the trigger window must belong to the same monitoring
        # session as t. Without this, a window at the very end of a session
        # can reach across a recording gap into the next session (which may
        # run at a different level) and fire a false FPT.
        if seg_start[t + consecutive - 1] != seg_start[t]:
            continue

        window = x[t:t + consecutive]

        beyond = window > upper_t[t]

        if two_sided:
            beyond = beyond | (window < lower_t[t])

        if beyond.all():
            fpt = t
            break

    result = {
        "fpt_index": fpt,
        "onset_index": fpt,
        "detection_index": None if fpt is None else fpt + consecutive - 1,
        "mu_t": mu_t,
        "upper_t": upper_t,
        "lower_t": lower_t,
        "seg_start": seg_start,
        # The healthy part before the FPT (inside its session) is the
        # baseline for the per-bearing feature normalization.
        "baseline_start": None if fpt is None else int(seg_start[fpt]),
        "baseline_end": fpt,
    }

    return result


# --------------------------------------------------
# 4. Labels that depend on FPT
# --------------------------------------------------

def add_fpt_labels(labels, fpt_index):

    out = labels.copy()

    idx = np.arange(len(out))
    rul_at_fpt = out["rul_hours"].iloc[fpt_index]

    # 0 = healthy stage, 1 = degradation stage (from the FPT on)
    out["stage"] = (idx >= fpt_index).astype(int)

    # Piecewise: flat cap before FPT, linear decrease after.
    out["rul_piecewise_hours"] = np.minimum(out["rul_hours"], rul_at_fpt)

    # Normalized from FPT (1 -> 0). Undefined (NaN) in the healthy stage.
    out["rul_norm_fpt"] = np.where(
        idx >= fpt_index,
        out["rul_hours"] / rul_at_fpt,
        np.nan
    )

    return out


# --------------------------------------------------
# 5. Plot for visual verification
# --------------------------------------------------

def plot_fpt(hours, rms, result, dataset, bearing, plot_dir):

    plt.figure(figsize=(12, 5))

    # Break the line across recording gaps (> 30 min) so that matplotlib
    # does not draw fake straight lines through missing data.
    gap = (hours.diff() * 60.0) > 30.0

    def broken(values):
        return pd.Series(np.asarray(values, dtype=float)).where(~gap)

    plt.plot(hours, broken(rms), linewidth=0.7, label="RMS (mean of channels)")
    plt.plot(hours, broken(result["mu_t"]), linestyle="--", linewidth=1,
             label="expanding mean")
    plt.plot(hours, broken(result["upper_t"]), linestyle=":", linewidth=1.2,
             label="mean + k*std (expanding)")

    # Mark the starts of new monitoring sessions (statistics restarts).
    session_starts = np.unique(result["seg_start"])

    for s in session_starts[session_starts > 0]:
        plt.axvline(hours.iloc[s], color="gray", linestyle="-.", linewidth=1,
                    label="session restart (gap)")

    if result["fpt_index"] is not None:
        plt.axvline(
            hours.iloc[result["fpt_index"]],
            color="red",
            linewidth=1.5,
            label=f"FPT (index {result['fpt_index']})"
        )

    # Remove duplicate legend entries.
    handles, labels_ = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels_, handles))
    plt.legend(unique.values(), unique.keys())

    plt.xlabel("Hours from start")
    plt.ylabel("RMS")
    plt.title(f"{dataset} - Bearing {bearing}: FPT detection")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_file = plot_dir / f"fpt_bearing_{bearing}.png"

    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved: {plot_file}")


# --------------------------------------------------
# 6. Process one dataset
# --------------------------------------------------

def process_dataset(dataset):

    print("\n====================================")
    print(f"RUL labels and FPT: {dataset}")
    print("====================================")

    f, t, hours = load_dataset(dataset)

    labels = make_base_labels(f, t, hours, dataset)

    labels.to_csv(
        PROCESSED / dataset / "rul_labels.csv",
        index=False,
        float_format="%.6f"
    )

    summary_rows = []

    for bearing, channels in TARGETS[dataset].items():

        rms_columns = [f"{ch}_rms" for ch in channels]

        missing = [c for c in rms_columns if c not in f.columns]

        if missing:
            raise KeyError(
                f"{dataset}: columns not found in features_all.csv: {missing}"
            )

        rms = f[rms_columns].mean(axis=1)

        result = detect_fpt(rms.to_numpy(), hours.to_numpy(), **FPT_CONFIG)

        print(f"\nBearing {bearing} (channels: {', '.join(channels)})")

        plot_fpt(
            hours,
            rms,
            result,
            dataset,
            bearing,
            PROCESSED / dataset / "plots" / "fpt"
        )

        if result["fpt_index"] is None:
            print("  FPT NOT detected with the current settings.")
            summary_rows.append({
                "dataset": dataset,
                "bearing": bearing,
                "n": len(labels),
                "T_end_h": hours.iloc[-1],
                "fpt_index": np.nan,
            })
            continue

        fpt_index = result["fpt_index"]

        bearing_labels = add_fpt_labels(labels, fpt_index)

        bearing_labels.to_csv(
            PROCESSED / dataset / f"rul_labels_bearing_{bearing}.csv",
            index=False,
            float_format="%.6f"
        )

        fpt_hours = hours.iloc[fpt_index]
        life_fraction = fpt_hours / hours.iloc[-1]

        print(
            f"  FPT index = {fpt_index}, {fpt_hours:.1f} h = "
            f"{life_fraction * 100:.1f}% of the run, "
            f"{len(labels) - fpt_index} post-FPT rows"
        )

        summary_rows.append({
            "dataset": dataset,
            "bearing": bearing,
            "n": len(labels),
            "T_end_h": hours.iloc[-1],
            "fpt_index": fpt_index,
            "fpt_hours": fpt_hours,
            "fpt_life_fraction": life_fraction,
            "post_fpt_rows": len(labels) - fpt_index,
            "rul_at_fpt_h": hours.iloc[-1] - fpt_hours,
            "mean_at_fpt": result["mu_t"][fpt_index],
            "upper_at_fpt": result["upper_t"][fpt_index],
        })

    return summary_rows


# --------------------------------------------------
# 7. Optional: FPT sensitivity table
#    python rul_labels.py --sensitivity
# --------------------------------------------------

SENSITIVITY_GRID = {
    "k": [3.0, 4.0],
    "consecutive": [2, 3, 5],
    "reset_gap_hours": [24.0, None],
    "two_sided": [False, True],
}


def run_sensitivity():
    """
    Shows how the detected FPT moves when the detector settings change.
    Cell = <FPT as share of the run's time> | <number of post-FPT rows>.
    "--" means no FPT was detected. Same settings for all bearings.
    """

    rows = []

    for dataset, bearing_map in TARGETS.items():

        f, t, hours = load_dataset(dataset)
        t_end = hours.iloc[-1]
        n = len(hours)

        for bearing, channels in bearing_map.items():

            bearing_id = f"T{dataset[0]}_B{bearing}"
            rms = f[[f"{ch}_rms" for ch in channels]].mean(axis=1).to_numpy()

            for k, consecutive, reset, two_sided in itertools.product(
                SENSITIVITY_GRID["k"],
                SENSITIVITY_GRID["consecutive"],
                SENSITIVITY_GRID["reset_gap_hours"],
                SENSITIVITY_GRID["two_sided"],
            ):

                config = dict(FPT_CONFIG)
                config.update(
                    k=k,
                    consecutive=consecutive,
                    reset_gap_hours=reset,
                    two_sided=two_sided,
                )

                result = detect_fpt(rms, hours.to_numpy(), **config)
                index = result["fpt_index"]

                cell = (
                    "--"
                    if index is None
                    else f"{hours.iloc[index] / t_end:.2f} | {n - index}"
                )

                reset_text = "off" if reset is None else f"{reset:g}h"

                rows.append({
                    "setting": (
                        f"k={k:g}, consec={consecutive}, "
                        f"reset={reset_text}, two_sided={two_sided}"
                    ),
                    "bearing": bearing_id,
                    "cell": cell,
                })

    table = (
        pd.DataFrame(rows)
        .pivot(index="setting", columns="bearing", values="cell")
    )

    print("\nFPT sensitivity (time share | post-FPT rows)")
    print("=" * 70)
    print(table.to_string())

    out = PROCESSED / "fpt_sensitivity.csv"
    table.to_csv(out)
    print(f"\nSaved: {out}")


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():

    all_rows = []

    for dataset in TARGETS:
        all_rows.extend(process_dataset(dataset))

    summary = pd.DataFrame(all_rows)

    summary_file = PROCESSED / "fpt_summary.csv"
    summary.to_csv(summary_file, index=False, float_format="%.4f")

    print("\n====================================")
    print("FPT SUMMARY")
    print("====================================")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved: {summary_file}")


if __name__ == "__main__":

    if "--sensitivity" in sys.argv:
        run_sensitivity()
    else:
        main()