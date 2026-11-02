from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROCESSED = Path(__file__).resolve().parent.parent / "Data" / "processed"

# Number of recordings documented in the IMS readme.
EXPECTED_N = {
    "1st_test": 2156,
    "2nd_test": 984,
    "3rd_test": 4448,
}

# Bearings with documented failure -> channels used for the RMS signal.
TARGETS = {
    "1st_test": {3: ["Bx_3", "By_3"], 4: ["Bx_4", "By_4"]},
    "2nd_test": {1: ["B1"]},
    "3rd_test": {3: ["B3"]},
}

# 3-sigma FPT detector settings (all are design choices, not "truth").
#   skip_initial_frac : share of the run skipped at the start (run-in transient)
#   baseline_frac     : share of the run used as the healthy reference,
#                       taken right after the skipped part
#   smooth_window     : causal (trailing) moving-average window, in recordings
#   k                 : threshold = mean +/- k * std of the smoothed baseline
#   consecutive       : number of consecutive points beyond the threshold to trigger
#   one_sided         : True = only upward excursions count (RMS grows with damage)
FPT_CONFIG = {
    "skip_initial_frac": 0.05,
    "baseline_frac": 0.10,
    "smooth_window": 25,
    "k": 3.0,
    "consecutive": 5,
    "min_baseline": 30,
    "one_sided": True,
}

TIME_FORMAT = "%Y.%m.%d.%H.%M.%S"


# --------------------------------------------------
# 2. Base RUL labels (no FPT needed)
# --------------------------------------------------

def load_dataset(dataset):
    """Load features, truncate to the documented length, build time axis."""

    f = pd.read_csv(PROCESSED / dataset / "features_all.csv")

    t = pd.to_datetime(
        f["recording"].astype(str),
        format=TIME_FORMAT
    )

    if not t.is_monotonic_increasing:
        raise ValueError(f"{dataset}: recordings are not time-ordered.")

    n = EXPECTED_N[dataset]
    f = f.iloc[:n].copy().reset_index(drop=True)
    t = t.iloc[:n].reset_index(drop=True)

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
# 3. FPT detection (causal 3-sigma, Robust-MBDL style)
# --------------------------------------------------

def detect_fpt(
    signal,
    baseline_frac=0.10,
    skip_initial_frac=0.05,
    smooth_window=25,
    k=3.0,
    consecutive=5,
    min_baseline=30,
    one_sided=True
):
    """
    Learning phase : mean and std from a healthy window near the start of the
                     run (after skipping the run-in transient).
    Detecting phase: scan forward; when `consecutive` smoothed points in a row
                     are beyond mean +/- k*std, the detector fires.

    Everything is causal: the smoothing is a trailing moving average and the
    baseline comes from the early part of the run only. The FPT is the index
    at which the trigger completes; the first point of that run is stored as
    `onset_index` for reference.
    """

    x = pd.Series(np.asarray(signal, dtype=float))

    smoothed = (
        x.rolling(smooth_window, min_periods=1)
        .mean()
        .to_numpy()
    )

    start = int(len(smoothed) * skip_initial_frac)
    n_base = max(int(len(smoothed) * baseline_frac), min_baseline)
    end = start + n_base

    if end >= len(smoothed):
        raise ValueError("Baseline window is longer than the signal.")

    mu = float(np.mean(smoothed[start:end]))
    sigma = float(np.std(smoothed[start:end]))

    lower = mu - k * sigma
    upper = mu + k * sigma

    if one_sided:
        outside = smoothed > upper
    else:
        outside = (smoothed > upper) | (smoothed < lower)

    result = {
        "fpt_index": None,
        "onset_index": None,
        "baseline_start": start,
        "baseline_end": end,
        "mu": mu,
        "sigma": sigma,
        "lower": lower,
        "upper": upper,
        "smoothed": smoothed,
    }

    run = 0

    for i in range(end, len(smoothed)):

        run = run + 1 if outside[i] else 0

        if run >= consecutive:
            result["fpt_index"] = i
            result["onset_index"] = i - consecutive + 1
            break

    return result


# --------------------------------------------------
# 4. Labels that depend on FPT
# --------------------------------------------------

def add_fpt_labels(labels, fpt_index):

    out = labels.copy()

    idx = np.arange(len(out))
    rul_at_fpt = out["rul_hours"].iloc[fpt_index]

    # 0 = healthy stage, 1 = degradation stage (after FPT)
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

    plt.plot(hours, rms, linewidth=0.6, alpha=0.4, label="RMS raw (mean of channels)")
    plt.plot(hours, result["smoothed"], linewidth=1.5, label="RMS smoothed (causal)")

    plt.axvspan(
        hours.iloc[result["baseline_start"]],
        hours.iloc[result["baseline_end"] - 1],
        alpha=0.15,
        label="baseline window"
    )

    plt.axhline(result["mu"], linestyle="--", linewidth=1, label="baseline mean")
    plt.axhline(result["upper"], linestyle=":", linewidth=1, label="upper threshold")

    if not FPT_CONFIG["one_sided"]:
        plt.axhline(result["lower"], linestyle=":", linewidth=1, label="lower threshold")

    if result["fpt_index"] is not None:
        plt.axvline(
            hours.iloc[result["fpt_index"]],
            color="red",
            linewidth=1.5,
            label=f"FPT (index {result['fpt_index']})"
        )

    plt.xlabel("Hours from start")
    plt.ylabel("RMS")
    plt.title(f"{dataset} - Bearing {bearing}: FPT detection")
    plt.legend()
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

        result = detect_fpt(rms.to_numpy(), **FPT_CONFIG)

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
            f"  FPT index = {fpt_index} (onset {result['onset_index']}), "
            f"{fpt_hours:.1f} h = {life_fraction * 100:.1f}% of the run"
        )

        summary_rows.append({
            "dataset": dataset,
            "bearing": bearing,
            "n": len(labels),
            "T_end_h": hours.iloc[-1],
            "fpt_index": fpt_index,
            "onset_index": result["onset_index"],
            "fpt_hours": fpt_hours,
            "fpt_life_fraction": life_fraction,
            "rul_at_fpt_h": hours.iloc[-1] - fpt_hours,
            "baseline_mean": result["mu"],
            "baseline_std": result["sigma"],
            "upper_threshold": result["upper"],
        })

    return summary_rows


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
    main()