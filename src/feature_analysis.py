from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import (
    load_recording,
    load_features,
    get_recording_files,
    VALID_RECORDINGS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS = ["1st_test", "2nd_test", "3rd_test"]
SAMPLING_FREQUENCY = 20_000
FEATURES_TO_PLOT = [
    "rms",
    "kurtosis",
    "crest_factor",
    "peak_to_peak",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "spectral_entropy",
]

CHANNEL_GROUPS = {
    "1st_test": [
        ["Bx_1", "By_1"],
        ["Bx_2", "By_2"],
        ["Bx_3", "By_3"],
        ["Bx_4", "By_4"],
    ],
    "2nd_test": [
        ["B1"],
        ["B2"],
        ["B3"],
        ["B4"],
    ],
    "3rd_test": [
        ["B1"],
        ["B2"],
        ["B3"],
        ["B4"],
    ],
}


def load_feature_table(dataset_name):

    # Valid recordings only (sensor-off files at the end are dropped).
    features = load_features(dataset_name)

    if "recording" not in features.columns:
        raise ValueError(
            f"'recording' column is missing in {dataset_name} features"
        )

    features["timestamp"] = pd.to_datetime(
        features["recording"],
        format="%Y.%m.%d.%H.%M.%S",
        errors="raise"
    )

    features = features.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return features

def plot_feature_trends(features, dataset_name, plot_dir):
    channel_groups = CHANNEL_GROUPS[dataset_name]
    for feature_name in FEATURES_TO_PLOT:
        fig, axes = plt.subplots(
            len(channel_groups),
            1,
            figsize=(14, 12),
            sharex=True
        )
        for index, ax in enumerate(axes):
            channels = channel_groups[index]
            plotted = False
            for channel in channels:
                column = f"{channel}_{feature_name}"
                if column not in features.columns:
                    continue
                ax.plot(
                    features["timestamp"],
                    features[column],
                    label=channel,
                    linewidth=0.8
                )
                plotted = True
            if dataset_name == "1st_test":
                title = f"Bearing {index + 1}"
            else:
                title = channels[0]
            ax.set_title(title)
            ax.set_ylabel(feature_name)
            ax.grid(True, alpha=0.3)
            if plotted:
                ax.legend()
        axes[-1].set_xlabel("Recording timestamp")
        fig.suptitle(
            f"{dataset_name}: "
            f"{feature_name.replace('_', ' ').title()} "
            "Across Experiment Lifetime"
        )
        fig.tight_layout()
        output_file = (
            plot_dir / f"{feature_name}_over_time.png"
        )
        fig.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )
        plt.close(fig)
        print(f"Saved feature trend: {output_file}")

def plot_frequency_spectra(
    files,
    dataset_name,
    plot_dir
):
    if not files:
        print(f"No recordings found in {dataset_name}.")
        return

    channel_groups = CHANNEL_GROUPS[dataset_name]

    selected_indices = sorted(
        set([
            0,
            len(files) // 2,
            len(files) - 1
        ])
    )

    for index in selected_indices:

        file = files[index]

        dataset = load_recording(
            dataset_name,
            file.name
        )

        fig, axes = plt.subplots(
            len(channel_groups),
            1,
            figsize=(14, 12),
            sharex=True
        )

        for group_index, ax in enumerate(axes):

            channels = channel_groups[group_index]

            for channel in channels:

                if channel not in dataset.columns:
                    continue

                signal = dataset[channel].to_numpy(
                    dtype=float
                )

                if not np.all(np.isfinite(signal)):
                    raise ValueError(
                        f"Invalid signal values in "
                        f"{dataset_name}, {file.name}, {channel}"
                    )

                if len(signal) == 0:
                    continue

                signal = signal - np.mean(signal)

                fft_values = np.fft.rfft(signal)

                frequencies = np.fft.rfftfreq(
                    len(signal),
                    d=1 / SAMPLING_FREQUENCY
                )

                # Relative amplitude spectrum.
                magnitude = (
                    np.abs(fft_values) / len(signal)
                )

                ax.plot(
                    frequencies,
                    magnitude,
                    label=channel,
                    linewidth=0.8
                )

            if dataset_name == "1st_test":
                title = f"Bearing {group_index + 1}"
            else:
                title = channels[0]

            ax.set_title(title)
            ax.set_ylabel("FFT magnitude")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel("Frequency (Hz)")

        fig.suptitle(
            f"{dataset_name}: Vibration Spectrum\n"
            f"Recording: {file.name}"
        )

        fig.tight_layout()

        output_file = (
            plot_dir
            / f"spectrum_recording_{index + 1}.png"
        )

        fig.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved frequency spectrum: {output_file}")

def analyze_dataset(dataset_name):

    print("\n====================================")
    print(f"Analyzing {dataset_name}")
    print("====================================")

    dataset_dir = PROCESSED_DIR / dataset_name

    feature_plot_dir = (
        dataset_dir / "plots" / "feature_trends"
    )

    spectrum_plot_dir = (
        dataset_dir / "plots" / "frequency_spectra"
    )

    feature_plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    spectrum_plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    features = load_feature_table(dataset_name)
    # Same valid range as the feature table, so the "last" spectrum is the
    # last real measurement and not a sensor-off file.
    files = get_recording_files(dataset_name)[
        :VALID_RECORDINGS[dataset_name]
    ]

    if len(features) != len(files):
        raise ValueError(
            f"Recording count mismatch in {dataset_name}: "
            f"{len(features)} feature rows vs "
            f"{len(files)} raw recordings."
        )

    print(f"Feature rows loaded: {len(features)}")
    print(f"Raw recordings found: {len(files)}")

    plot_feature_trends(
        features,
        dataset_name,
        feature_plot_dir
    )

    plot_frequency_spectra(
        files,
        dataset_name,
        spectrum_plot_dir
    )

    print(f"\nCompleted analysis of {dataset_name}.")


def main():

    for dataset_name in DATASETS:
        analyze_dataset(dataset_name)

    print("\nAll dataset visualizations completed.")


if __name__ == "__main__":
    main()
