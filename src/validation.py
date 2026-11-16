# validation.py
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

DATASETS = ["1st_test", "2nd_test", "3rd_test"]
HI_FILENAME = "health_indicators.csv"
RESULTS_FILENAME = "health_indicator_validation.csv"

ALL_RESULTS_FILE = PROCESSED_DIR / "health_indicator_validation_all.csv"

COLUMN_ORDER = [
    "dataset",
    "bearing",
    "version",
    "n_observations",
    "HI_min",
    "HI_max",
    "HI_range",
    "endpoint_change",
    "monotonicity",
    "spearman_rho",
    "positive_steps",
    "negative_steps",
    "normalized_roughness",
]

def evaluate_indicator(values):

    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)
    time = np.arange(len(values), dtype=float)[valid]
    hi = values[valid]

    if len(hi) < 3:
        return None

    differences = np.diff(hi)

    positive = int(np.sum(differences > 0))
    negative = int(np.sum(differences < 0))
    nonzero = positive + negative

    monotonicity = (
        (positive - negative) / nonzero
        if nonzero > 0
        else 0.0
    )

    spearman_rho = spearmanr(time, hi).statistic

    hi_min = float(np.min(hi))
    hi_max = float(np.max(hi))
    hi_range = hi_max - hi_min

    normalized_roughness = (
        float(np.mean(np.abs(differences)) / hi_range)
        if hi_range > 0
        else np.nan
    )

    endpoint_change = float(hi[-1] - hi[0])

    return {
        "n_observations": len(hi),
        "HI_min": hi_min,
        "HI_max": hi_max,
        "HI_range": hi_range,
        "endpoint_change": endpoint_change,
        "monotonicity": float(monotonicity),
        "spearman_rho": float(spearman_rho),
        "positive_steps": positive,
        "negative_steps": negative,
        "normalized_roughness": normalized_roughness,
    }

def evaluate_dataset(dataset_name):

    dataset_dir = PROCESSED_DIR / dataset_name
    hi_file = dataset_dir / HI_FILENAME
    results_file = dataset_dir / RESULTS_FILENAME

    print("\n====================================")
    print(f"HI validation: {dataset_name}")
    print("====================================")

    if not hi_file.exists():
        print(
            f"Health indicator file not found: {hi_file}\n"
            "Run health_indicator.py first. Skipping this dataset."
        )
        return pd.DataFrame()

    data = pd.read_csv(hi_file)

    all_results = []

    for bearing_number in range(1, 5):

        for version, column in [
            ("raw", f"Bearing_{bearing_number}_HI_raw"),
            ("smoothed", f"Bearing_{bearing_number}_HI_smoothed"),
        ]:

            if column not in data.columns:
                print(f"Missing column: {column}. Skipping.")
                continue

            metrics = evaluate_indicator(data[column].to_numpy())

            if metrics is None:
                print(
                    f"Bearing {bearing_number} ({version}): "
                    "fewer than 3 valid values. Skipping."
                )
                continue

            metrics["dataset"] = dataset_name
            metrics["bearing"] = bearing_number
            metrics["version"] = version

            all_results.append(metrics)

    if not all_results:
        print(f"No health indicators could be evaluated for {dataset_name}.")
        return pd.DataFrame()

    results = pd.DataFrame(all_results)[COLUMN_ORDER]

    results.to_csv(
        results_file,
        index=False,
        float_format="%.6f",
    )

    print(
        results.drop(columns="dataset").to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(f"\nResults saved to: {results_file}")

    return results


def main():

    frames = []

    for dataset_name in DATASETS:
        result = evaluate_dataset(dataset_name)

        if not result.empty:
            frames.append(result)

    if not frames:
        raise ValueError(
            "No health indicators were available for evaluation. "
            "Run health_indicator.py first."
        )

    combined = pd.concat(frames, ignore_index=True)

    combined.to_csv(
        ALL_RESULTS_FILE,
        index=False,
        float_format="%.6f",
    )
    print(f"\nCombined results saved to: {ALL_RESULTS_FILE}")

if __name__ == "__main__":
    main()