import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rul_labels import (
    PROCESSED,
    TARGETS,
    FPT_CONFIG,
    load_dataset,
    make_base_labels,
    detect_fpt,
    add_fpt_labels,
)


EXCLUDE_SUFFIXES = (
    "_target_hz",
    "_theoretical_hz",
    "_bins_in_band",
)

OUT_RAW = PROCESSED / "ml_dataset_raw.csv"
OUT_Z = PROCESSED / "ml_dataset_zscore.csv"
OUT_FEATURES = PROCESSED / "ml_features_used.csv"
LABEL_PLOT = PROCESSED / "plots" / "label_check.png"

META_COLUMNS = [
    "bearing_id",
    "test_group",
    "recording",
    "hours_from_start",
    "stage",
    "rul_hours",
    "rul_norm_total",
    "rul_piecewise_hours",
    "rul_norm_fpt",
]


def common_feature_table(f, channels):

    per_channel = {}

    for ch in channels:

        prefix = f"{ch}_"

        per_channel[ch] = {
            column[len(prefix):]: column
            for column in f.columns
            if column.startswith(prefix)
        }
    names = set.intersection(
        *[set(mapping) for mapping in per_channel.values()]
    )

    names = sorted(
        name for name in names
        if not name.endswith(EXCLUDE_SUFFIXES)
    )

    table = {}

    for name in names:

        parts = pd.concat(
            [
                pd.to_numeric(f[per_channel[ch][name]], errors="coerce")
                for ch in channels
            ],
            axis=1
        )

        table[name] = parts.mean(axis=1)

    return pd.DataFrame(table)


def baseline_zscore(features, start, end):

    baseline = features.iloc[start:end]

    mu = baseline.mean()
    sigma = baseline.std(ddof=0)

    floor = 1e-3 * mu.abs() + 1e-12
    sigma = np.maximum(sigma, floor)

    return (features - mu) / sigma


def make_folds(meta):
    ids = (
        meta[["bearing_id", "test_group"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    folds = []

    for group in ids["test_group"].unique():

        test_ids = ids.loc[ids["test_group"] == group, "bearing_id"].tolist()
        train_ids = ids.loc[ids["test_group"] != group, "bearing_id"].tolist()

        folds.append({
            "name": f"hold_out_{group}",
            "train_ids": train_ids,
            "test_ids": test_ids,
        })

    return folds


def build():

    bearings = []

    for dataset, bearing_map in TARGETS.items():

        f, t, hours = load_dataset(dataset)
        base_labels = make_base_labels(f, t, hours, dataset)

        test_group = f"test{dataset[0]}"

        for bearing, channels in bearing_map.items():

            bearing_id = f"T{dataset[0]}_B{bearing}"

            rms = f[[f"{ch}_rms" for ch in channels]].mean(axis=1)
            result = detect_fpt(rms.to_numpy(), hours.to_numpy(), **FPT_CONFIG)

            if result["fpt_index"] is None:
                print(f"  {bearing_id}: FPT not detected -> EXCLUDED.")
                continue

            labels = add_fpt_labels(base_labels, result["fpt_index"])
            labels.insert(0, "test_group", test_group)
            labels.insert(0, "bearing_id", bearing_id)

            features = common_feature_table(f, channels)

            bearings.append({
                "id": bearing_id,
                "labels": labels,
                "features": features,
                "start": result["baseline_start"],
                "end": result["baseline_end"],
            })

            print(
                f"  {bearing_id}: FPT index {result['fpt_index']}, "
                f"{int(labels['stage'].sum())} post-FPT rows, "
                f"{features.shape[1]} candidate features."
            )

    if len(bearings) < 2:
        raise RuntimeError(
            "Fewer than two bearings with a detected FPT. "
            "Fix the FPT step first."
        )

    common = set.intersection(*[set(b["features"].columns) for b in bearings])

    dropped = {}
    valid = []

    for name in sorted(common):

        reason = None

        for b in bearings:

            col = b["features"][name]

            if not np.isfinite(col).all():
                reason = f"NaN/inf in {b['id']}"
                break

            if col.nunique() <= 1:
                reason = f"constant in {b['id']}"
                break

        if reason:
            dropped[name] = reason
        else:
            valid.append(name)

    print(f"\nUnified features kept: {len(valid)}")
    print(f"Dropped (not valid in every bearing): {len(dropped)}")

    for name, reason in dropped.items():
        print(f"  - {name}: {reason}")

    raw_frames = []
    z_frames = []

    for b in bearings:

        raw = b["features"][valid]
        z = baseline_zscore(raw, b["start"], b["end"])

        raw_frames.append(pd.concat([b["labels"], raw], axis=1))
        z_frames.append(pd.concat([b["labels"], z], axis=1))

    raw_df = pd.concat(raw_frames, ignore_index=True)
    z_df = pd.concat(z_frames, ignore_index=True)

    raw_df = raw_df[META_COLUMNS + valid]
    z_df = z_df[META_COLUMNS + valid]

    raw_df.to_csv(OUT_RAW, index=False, float_format="%.6f")
    z_df.to_csv(OUT_Z, index=False, float_format="%.6f")
    pd.Series(valid, name="feature").to_csv(OUT_FEATURES, index=False)

    print(f"\nSaved: {OUT_RAW}")
    print(f"Saved: {OUT_Z}")
    print(f"Saved: {OUT_FEATURES}")

    return z_df


def verify_labels(df):

    all_ok = True

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (bearing, g) in zip(axes.ravel(), df.groupby("bearing_id")):

        post = g[g["stage"] == 1]

        checks = {
            "rul_hours decreasing": g["rul_hours"].is_monotonic_decreasing,
            "rul_hours ends at 0": bool(np.isclose(g["rul_hours"].iloc[-1], 0.0)),
            "rul_norm_fpt starts at 1": bool(np.isclose(post["rul_norm_fpt"].iloc[0], 1.0)),
            "rul_norm_fpt ends at 0": bool(np.isclose(post["rul_norm_fpt"].iloc[-1], 0.0)),
            "rul_norm_fpt decreasing": post["rul_norm_fpt"].is_monotonic_decreasing,
            "no NaN after FPT": not post["rul_norm_fpt"].isna().any(),
            "piecewise flat before FPT": g.loc[g["stage"] == 0, "rul_piecewise_hours"].nunique() <= 1,
        }

        failed = [name for name, ok in checks.items() if not ok]
        all_ok = all_ok and not failed

        print(
            f"  {bearing}: "
            + ("all label checks PASS" if not failed else f"FAIL -> {failed}")
        )

        ax.plot(g["hours_from_start"], g["rul_hours"], label="rul_hours")
        ax.plot(g["hours_from_start"], g["rul_piecewise_hours"], label="rul_piecewise_hours")

        ax2 = ax.twinx()
        ax2.plot(g["hours_from_start"], g["rul_norm_fpt"], color="green")
        ax2.set_ylim(-0.05, 1.05)

        ax.set_title(bearing)
        ax.set_xlabel("hours from start")
        ax.set_ylabel("hours")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    LABEL_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(LABEL_PLOT, dpi=130)
    plt.close()

    print(f"  Label plot saved: {LABEL_PLOT}")
    print("  LABELS OK" if all_ok else "  LABEL PROBLEM FOUND, see above")


def main():

    print("Building unified dataset...")
    df = build()

    print("\nLabel verification:")
    verify_labels(df)

    print("\nPost-FPT rows per bearing (these are the rows models use):")
    print(
        df[df["stage"] == 1]
        .groupby("bearing_id")
        .size()
        .to_string()
    )

    print("\nFolds (leave-one-test-out):")

    for fold in make_folds(df):
        print(
            f"  {fold['name']}: train={fold['train_ids']} "
            f"-> test={fold['test_ids']}"
        )


if __name__ == "__main__":
    main()