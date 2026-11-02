from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

DATASETS = ["1st_test", "2nd_test", "3rd_test"]

LOESS_SPAN = 0.1
TOP_N_PER_CHANNEL = 5

SCORE_COLUMNS = [
    "monotonicity",
    "trendability",
    "robustness",
]

# Columns that are NOT measured from the signal. They are fixed
# parameters (target frequencies, bin counts) and never change over
# time, so they must never be used as health-indicator candidates.
EXCLUDE_SUFFIXES = (
    "_target_hz",
    "_theoretical_hz",
    "_bins_in_band",
)

CHANNELS = {
    "1st_test": [
        "Bx_1", "By_1",
        "Bx_2", "By_2",
        "Bx_3", "By_3",
        "Bx_4", "By_4",
    ],
    "2nd_test": ["B1", "B2", "B3", "B4"],
    "3rd_test": ["B1", "B2", "B3", "B4"],
}


# --------------------------------------------------
# 2. Group channels by physical bearing
# --------------------------------------------------

def get_bearing_channels(dataset_name):

    if dataset_name == "1st_test":
        return {
            1: ["Bx_1", "By_1"],
            2: ["Bx_2", "By_2"],
            3: ["Bx_3", "By_3"],
            4: ["Bx_4", "By_4"],
        }

    return {
        1: ["B1"],
        2: ["B2"],
        3: ["B3"],
        4: ["B4"],
    }


# --------------------------------------------------
# 3. Feature selection
# --------------------------------------------------

def compare_and_select_features(
    quality,
    channels,
    features,
    top_n=TOP_N_PER_CHANNEL
):
    """
    Rank features separately within each channel.

    Before ranking, features that are not real signal measurements
    are removed:
      - fixed parameters (names ending in EXCLUDE_SUFFIXES)
      - features missing from the feature table
      - constant features (only one unique value)

    The composite score is the mean of available quality
    metrics. At least two metrics must be available.
    """

    results = quality.copy()

    required_columns = [
        "feature",
        "monotonicity",
        "trendability",
        "robustness",
    ]

    missing = [
        column
        for column in required_columns
        if column not in results.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in feature-quality file: {missing}"
        )

    # Retain only valid channel-feature combinations.
    results = results[
        results["channel"].isin(channels)
    ].copy()

    # Remove fixed parameters and constant features.
    def is_real_signal_feature(name):

        if str(name).endswith(EXCLUDE_SUFFIXES):
            return False

        if name not in features.columns:
            return False

        column = pd.to_numeric(
            features[name],
            errors="coerce"
        )

        return column.nunique() > 1

    before = len(results)

    results = results[
        results["feature"].apply(is_real_signal_feature)
    ].copy()

    print(
        f"  Removed {before - len(results)} non-signal / "
        f"constant features, {len(results)} remain."
    )

    # Count the available quality metrics for each feature.
    results["available_metrics"] = (
        results[SCORE_COLUMNS].notna().sum(axis=1)
    )

    # Require at least two available metrics.
    results["eligible"] = (
        results["available_metrics"] >= 2
    )

    # Equal-weight average of available metrics.
    results["composite_score"] = (
        results[SCORE_COLUMNS].mean(axis=1, skipna=True)
    )

    results["selected"] = False

    eligible = results[results["eligible"]]

    # Select the top N features independently per channel.
    for channel, group in eligible.groupby(
        "channel",
        sort=False
    ):

        top_indices = (
            group.sort_values(
                ["composite_score", "feature"],
                ascending=[False, True]
            )
            .head(top_n)
            .index
        )

        results.loc[top_indices, "selected"] = True

    channel_order = {
        channel: index
        for index, channel in enumerate(channels)
    }

    results["channel_order"] = (
        results["channel"].map(channel_order)
    )

    results = results.sort_values(
        [
            "channel_order",
            "composite_score",
            "feature"
        ],
        ascending=[True, False, True],
        na_position="last"
    )

    results = results.drop(
        columns="channel_order"
    )

    selected = results[
        results["selected"]
    ].copy()

    return results.reset_index(drop=True), selected.reset_index(drop=True)


# --------------------------------------------------
# 4. Construct one Health Indicator
# --------------------------------------------------

def construct_health_indicator(
    features,
    selected_feature_names,
    loess_span=LOESS_SPAN
):
    """
    Construct an exploratory HI from selected features.

    Steps:
    1. Min-max normalize each selected feature.
    2. Align feature direction with recording time.
    3. Calculate the equal-weight average.
    4. Smooth the combined HI using LOWESS.

    Every skipped feature prints the reason.
    """

    selected_feature_names = [
        name
        for name in selected_feature_names
        if name in features.columns
    ]

    if not selected_feature_names:
        raise ValueError(
            "No selected features were found in the feature table."
        )

    aligned_features = pd.DataFrame(
        index=features.index
    )

    used_features = []

    time_index = np.arange(
        len(features),
        dtype=float
    )

    for name in selected_feature_names:

        values = pd.to_numeric(
            features[name],
            errors="coerce"
        ).to_numpy(dtype=float)

        finite = np.isfinite(values)

        if finite.sum() < 2:
            print(f"  SKIP {name}: finite values = {finite.sum()}")
            continue

        minimum = np.nanmin(values)
        maximum = np.nanmax(values)

        # Constant features cannot indicate degradation.
        if np.isclose(maximum, minimum):
            print(
                f"  SKIP {name}: constant "
                f"(min={minimum}, max={maximum})"
            )
            continue

        # Min-max normalization.
        normalized = (
            values - minimum
        ) / (
            maximum - minimum
        )

        valid = np.isfinite(normalized)

        if np.std(normalized[valid]) == 0:
            print(f"  SKIP {name}: std = 0")
            continue

        correlation = np.corrcoef(
            normalized[valid],
            time_index[valid]
        )[0, 1]

        if not np.isfinite(correlation):
            print(f"  SKIP {name}: correlation is NaN")
            continue

        # Align direction with the observed temporal trend.
        if correlation < 0:
            normalized = 1.0 - normalized

        aligned_features[name] = normalized

        used_features.append(name)

    if not used_features:
        raise ValueError(
            "No usable selected features remained after preprocessing."
        )

    # Equal-weight fusion of normalized features.
    raw_hi = aligned_features.mean(
        axis=1,
        skipna=True
    )

    # Interpolate missing values before smoothing.
    raw_hi = raw_hi.interpolate(
        method="linear",
        limit_direction="both"
    )

    if raw_hi.isna().any():
        raise ValueError(
            "The combined HI contains missing values."
        )

    # LOWESS smoothing.
    smoothed_hi = lowess(
        endog=raw_hi.to_numpy(),
        exog=time_index,
        frac=loess_span,
        return_sorted=False
    )

    result = pd.DataFrame({
        "HI_raw": raw_hi,
        "HI_smoothed": smoothed_hi,
    })

    return result, used_features


# --------------------------------------------------
# 5. Plot Health Indicator
# --------------------------------------------------

def plot_health_indicator(
    hi,
    dataset_name,
    bearing_number,
    plot_dir
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        hi.index,
        hi["HI_raw"],
        label="Raw HI",
        alpha=0.5
    )

    plt.plot(
        hi.index,
        hi["HI_smoothed"],
        label="LOWESS-smoothed HI",
        linewidth=2
    )

    plt.xlabel("Recording index")
    plt.ylabel("Health Indicator")

    plt.title(
        f"{dataset_name} - Bearing {bearing_number}: "
        "Health Indicator"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = (
        plot_dir
        / f"health_indicator_bearing_{bearing_number}.png"
    )

    plt.savefig(
        plot_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved: {plot_file}")


# --------------------------------------------------
# 6. Process one experiment
# --------------------------------------------------

def process_dataset(dataset_name):

    print("\n====================================")
    print(f"Health indicator construction: {dataset_name}")
    print("====================================")

    dataset_dir = PROCESSED_DIR / dataset_name

    quality_file = dataset_dir / "feature_quality.csv"
    features_file = dataset_dir / "features_all.csv"

    comparison_file = (
        dataset_dir / "feature_comparison.csv"
    )

    selected_file = (
        dataset_dir / "selected_features.csv"
    )

    hi_file = (
        dataset_dir / "health_indicators.csv"
    )

    plots_dir = (
        dataset_dir / "plots" / "health_indicators"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check input files.
    for file_path in [quality_file, features_file]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load feature-quality results and extracted features.
    quality = pd.read_csv(quality_file)
    features = pd.read_csv(features_file)

    channels = CHANNELS[dataset_name]

    if len(quality) == 0 or len(features) == 0:
        raise ValueError(
            f"Empty input table found in {dataset_name}."
        )

    # --------------------------------------------------
    # Feature selection
    # --------------------------------------------------

    comparison, selected = compare_and_select_features(
        quality,
        channels,
        features
    )

    comparison.to_csv(
        comparison_file,
        index=False,
        float_format="%.6f"
    )

    selected.to_csv(
        selected_file,
        index=False,
        float_format="%.6f"
    )

    print(f"Feature comparison saved: {comparison_file}")
    print(f"Selected features saved: {selected_file}")

    if selected.empty:
        raise ValueError(
            f"No features were selected for {dataset_name}."
        )

    # --------------------------------------------------
    # Recording metadata
    # --------------------------------------------------

    if "recording" in features.columns:

        recording_column = features["recording"]

    elif "timestamp" in features.columns:

        recording_column = features["timestamp"]

    elif "recording_index" in features.columns:

        recording_column = features["recording_index"]

    else:

        recording_column = pd.Series(
            np.arange(len(features)),
            name="recording_index"
        )

    output = pd.DataFrame({
        "recording": recording_column
    })

    # --------------------------------------------------
    # Construct HI for each physical bearing
    # --------------------------------------------------

    bearing_channels = get_bearing_channels(
        dataset_name
    )

    for bearing_number, channels_for_bearing in (
        bearing_channels.items()
    ):

        bearing_features = selected[
            selected["channel"].isin(channels_for_bearing)
        ]["feature"].tolist()

        if not bearing_features:

            print(
                f"Bearing {bearing_number}: "
                "no selected features found."
            )

            output[
                f"Bearing_{bearing_number}_HI_raw"
            ] = np.nan

            output[
                f"Bearing_{bearing_number}_HI_smoothed"
            ] = np.nan

            continue

        print(f"\nBearing {bearing_number}:")

        hi, used_features = construct_health_indicator(
            features=features,
            selected_feature_names=bearing_features
        )

        output[
            f"Bearing_{bearing_number}_HI_raw"
        ] = hi["HI_raw"]

        output[
            f"Bearing_{bearing_number}_HI_smoothed"
        ] = hi["HI_smoothed"]

        plot_health_indicator(
            hi,
            dataset_name,
            bearing_number,
            plots_dir
        )

        print(
            f"Bearing {bearing_number}: "
            f"{len(used_features)} features used."
        )

        for feature in used_features:
            print(f"  - {feature}")

    # --------------------------------------------------
    # Save all Health Indicators
    # --------------------------------------------------

    output.to_csv(
        hi_file,
        index=False,
        float_format="%.6f"
    )

    print(f"\nHealth indicators saved: {hi_file}")


# --------------------------------------------------
# 7. Main
# --------------------------------------------------

def main():

    for dataset_name in DATASETS:
        process_dataset(dataset_name)

    print("\nAll health indicators constructed successfully.")


if __name__ == "__main__":
    main()
