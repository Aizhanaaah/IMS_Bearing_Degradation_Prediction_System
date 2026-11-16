from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from data_loader import load_features

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS = ["1st_test", "2nd_test", "3rd_test"]

CHANNELS = {
    "1st_test": [
        "Bx_1", "By_1", "Bx_2", "By_2",
        "Bx_3", "By_3", "Bx_4", "By_4"
    ],
    "2nd_test": ["B1", "B2", "B3", "B4"],
    "3rd_test": ["B1", "B2", "B3", "B4"]
}

FAILED_BEARINGS = {
    "1st_test": [3, 4],
    "2nd_test": [1],
    "3rd_test": [3]
}
ROBUSTNESS_WINDOW = 21

def calculate_monotonicity(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    differences = np.diff(values)

    increasing = np.sum(differences > 0)
    decreasing = np.sum(differences < 0)

    return float(
        abs(increasing - decreasing) / (len(values) - 1)
    )

def calculate_robustness(values, window=ROBUSTNESS_WINDOW):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return np.nan
    series = pd.Series(values)
    smoothed = series.rolling(
        window=min(window, len(series)),
        center=True,
        min_periods=5
    ).median()
    deviations = (series - smoothed).dropna()
    trend = smoothed.dropna()
    valid = trend.index[
        series.loc[trend.index].abs() > 1e-12
    ]

    if len(valid) == 0:
        return np.nan
    original = series.loc[valid]
    residuals = deviations.loc[valid]
    normalized_residuals = (
        residuals.abs() / original.abs()
    )

    robustness = np.exp(-normalized_residuals).mean()
    return float(robustness)

def calculate_pearson_correlation(values_a, values_b):

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if len(a) != len(b):
        return np.nan

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return np.nan

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])

def identify_channel_and_feature(column, channels):

    for channel in sorted(channels, key=len, reverse=True):

        prefix = channel + "_"

        if column.startswith(prefix):
            feature_name = column[len(prefix):]
            return channel, feature_name

    return None, None


def get_bearing_number(channel):

    if "_" in channel:
        return int(channel.split("_")[-1])

    return int(channel[1:])


def get_sensor_group(channel):

    if channel.startswith("Bx_"):
        return "Bx"

    if channel.startswith("By_"):
        return "By"

    return "B"

def calculate_trendability(
    features,
    feature_name,
    sensor_group,
    channels,
    failed_bearings
):
    columns = []
    for channel in channels:
        bearing_number = get_bearing_number(channel)
        if bearing_number not in failed_bearings:
            continue
        if get_sensor_group(channel) != sensor_group:
            continue
        column = f"{channel}_{feature_name}"
        if column in features.columns:
            columns.append(column)
    if len(columns) < 2:
        return np.nan, np.nan
    correlations = []
    for column_a, column_b in combinations(columns, 2):
        correlation = calculate_pearson_correlation(
            features[column_a],
            features[column_b]
        )
        if np.isfinite(correlation):
            correlations.append(correlation)
    if not correlations:
        return np.nan, np.nan
    trendability = min(abs(r) for r in correlations)
    direction_agreement = np.mean(
        [r > 0 for r in correlations]
    )
    return float(trendability), float(direction_agreement)

def evaluate_features(features, dataset_name):

    channels = CHANNELS[dataset_name]
    failed_bearings = FAILED_BEARINGS[dataset_name]

    excluded = {
        "timestamp",
        "recording",
        "recording_index",
        "dataset"
    }

    feature_columns = [
        column
        for column in features.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    results = []

    for column in feature_columns:

        channel, feature_name = identify_channel_and_feature(
            column,
            channels
        )

        if channel is None:
            continue

        values = features[column].to_numpy(dtype=float)

        sensor_group = get_sensor_group(channel)

        trendability, direction_agreement = (
            calculate_trendability(
                features,
                feature_name,
                sensor_group,
                channels,
                failed_bearings
            )
        )

        results.append({
            "channel": channel,
            "feature_name": feature_name,
            "feature": column,
            "monotonicity": calculate_monotonicity(values),
            "trendability": trendability,
            "direction_agreement": direction_agreement,
            "robustness": calculate_robustness(values)
        })

    results = pd.DataFrame(results)
    if results.empty:
        raise ValueError(
            f"No valid feature columns found in {dataset_name}."
        )
    channel_order = {
        channel: index
        for index, channel in enumerate(channels)
    }
    results["channel_order"] = results["channel"].map(
        channel_order
    )
    results = results.sort_values(
        ["channel_order", "feature_name"]
    )
    results = results.drop(columns="channel_order")
    return results.reset_index(drop=True)

def main():

    for dataset_name in DATASETS:

        print("\n====================================")
        print(f"Feature evaluation: {dataset_name}")
        print("====================================")

        dataset_dir = PROCESSED_DIR / dataset_name

        input_file = dataset_dir / "features_all.csv"
        output_file = dataset_dir / "feature_quality.csv"

        if not input_file.exists():
            print(f"File not found: {input_file}")
            print("Skipping dataset.")
            continue

        features = load_features(dataset_name)

        results = evaluate_features(
            features,
            dataset_name
        )

        results.to_csv(
            output_file,
            index=False,
            float_format="%.3f"
        )

        print("\nFeature quality results:")
        print(results.to_string(index=False))

if __name__ == "__main__":
    main()
