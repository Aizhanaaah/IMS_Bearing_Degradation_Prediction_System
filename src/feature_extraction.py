from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_recording

DATASETS = [
    "1st_test",
    "2nd_test",
    "3rd_test",
]

SAMPLING_FREQUENCY = 20_000

BEARING_PARAMETERS = {
    "number_of_rolling_elements": 16,
    "rolling_element_diameter_m": 0.0084,
    "pitch_diameter_m": 0.0715,
    "contact_angle_deg": 15.17,
    "shaft_frequency_hz": 2000 / 60,
}

FREQUENCY_BANDWIDTH_HZ = 2.0
NUMBER_OF_HARMONICS = 3
PLOT_EVERY_RECORDING = False

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data"
PROCESSED_DIR = DATA_DIR / "processed"


def calculate_bearing_frequencies(parameters):
    required = [
        "number_of_rolling_elements",
        "rolling_element_diameter_m",
        "pitch_diameter_m",
        "contact_angle_deg",
        "shaft_frequency_hz",
    ]

    if any(parameters.get(key) is None for key in required):
        return None

    nb = float(parameters["number_of_rolling_elements"])
    d = float(parameters["rolling_element_diameter_m"])
    D = float(parameters["pitch_diameter_m"])

    phi = np.deg2rad(
        float(parameters["contact_angle_deg"])
    )

    fr = float(parameters["shaft_frequency_hz"])

    if nb <= 0 or d <= 0 or D <= 0 or fr <= 0 or d >= D:
        raise ValueError("Invalid bearing parameters.")

    ratio = (d / D) * np.cos(phi)

    return {
        "FTF": (fr / 2.0) * (1.0 - ratio),

        "BPFO": (nb * fr / 2.0) * (1.0 - ratio),

        "BPFI": (nb * fr / 2.0) * (1.0 + ratio),

        "BSF": (
            D * fr / (2.0 * d)
        ) * (1.0 - ratio**2),
    }


def measure_frequency_band(
    frequencies,
    amplitude,
    center_hz,
    half_width_hz
):
    mask = (
        (frequencies >= center_hz - half_width_hz)
        &
        (frequencies <= center_hz + half_width_hz)
    )

    if not np.any(mask):

        return {
            "peak_amplitude": np.nan,
            "mean_amplitude": np.nan,
            "band_power": np.nan,
            "bins_in_band": 0,
        }

    band_amplitude = amplitude[mask]

    return {
        "peak_amplitude": float(np.max(band_amplitude)),
        "mean_amplitude": float(np.mean(band_amplitude)),
        "band_power": float(np.sum(band_amplitude**2)),
        "bins_in_band": int(np.sum(mask)),
    }

def extract_features(
    dataset,
    recording_label="recording",
    dataset_name="unknown",
    recording_index=0,
    plot_path=None
):

    features = {

        "dataset": dataset_name,

        "recording_index": recording_index,

        "recording": recording_label,
    }

    fault_frequencies = calculate_bearing_frequencies(
        BEARING_PARAMETERS
    )

    plot_signals = []

    for column in dataset.columns:

        signal = pd.to_numeric(
            dataset[column],
            errors="coerce"
        )
        values = signal.to_numpy(dtype=float)

        if len(values) == 0:
            raise ValueError(
                f"Empty signal: {dataset_name}, {column}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite measurements detected in "
                f"{dataset_name}, {recording_label}, {column}."
            )

        mean_abs = np.mean(np.abs(values))
        rms = np.sqrt(np.mean(values**2))
        peak = np.max(np.abs(values))
        features[f"{column}_mean"] = np.mean(values)
        features[f"{column}_abs_mean"] = mean_abs
        features[f"{column}_rms"] = rms
        features[f"{column}_std"] = np.std(values)
        features[f"{column}_variance"] = np.var(values)
        features[f"{column}_skewness"] = signal.skew()
        features[f"{column}_kurtosis"] = signal.kurtosis()
        features[f"{column}_max"] = np.max(values)
        features[f"{column}_min"] = np.min(values)
        features[f"{column}_peak_to_peak"] = (
            np.max(values) - np.min(values)
        )
        features[f"{column}_crest_factor"] = (
            peak / rms if rms != 0 else np.nan
        )
        features[f"{column}_shape_factor"] = (
            rms / mean_abs if mean_abs != 0 else np.nan
        )
        features[f"{column}_impulse_factor"] = (
            peak / mean_abs if mean_abs != 0 else np.nan
        )
        centered_signal = values - np.mean(values)
        fft_values = np.fft.rfft(centered_signal)
        magnitude_all = np.abs(fft_values)
        frequencies_all = np.fft.rfftfreq(
            len(centered_signal),
            d=1.0 / SAMPLING_FREQUENCY
        )
        frequencies = frequencies_all[1:]
        magnitude = magnitude_all[1:]
        if len(magnitude) > 0 and np.sum(magnitude) > 0:
            dominant_index = np.argmax(magnitude)
            dominant_frequency = frequencies[dominant_index]
            spectral_centroid = (
                np.sum(frequencies * magnitude)
                / np.sum(magnitude)
            )
            spectral_bandwidth = np.sqrt(
                np.sum(
                    ((frequencies - spectral_centroid) ** 2)
                    * magnitude
                )
                / np.sum(magnitude)
            )
            power = magnitude**2
            probabilities = power / np.sum(power)
            entropy = -np.sum(
                probabilities * np.log2(probabilities + 1e-12)
            )
            spectral_entropy = (
                entropy / np.log2(len(probabilities))
                if len(probabilities) > 1
                else 0.0
            )

        else:

            dominant_frequency = 0.0
            spectral_centroid = 0.0
            spectral_bandwidth = 0.0
            spectral_entropy = 0.0

        features[
            f"{column}_dominant_frequency_hz"
        ] = dominant_frequency

        features[
            f"{column}_spectral_centroid_hz"
        ] = spectral_centroid

        features[
            f"{column}_spectral_bandwidth_hz"
        ] = spectral_bandwidth

        features[
            f"{column}_spectral_entropy"
        ] = spectral_entropy
        if fault_frequencies is None:
            for fault_name in ("FTF", "BPFO", "BPFI", "BSF"):
                features[
                    f"{column}_{fault_name}_theoretical_hz"
                ] = np.nan
                for harmonic in range(
                    1,
                    NUMBER_OF_HARMONICS + 1
                ):
                    prefix = (
                        f"{column}_{fault_name}_H{harmonic}"
                    )
                    features[
                        f"{prefix}_peak_amplitude"
                    ] = np.nan
                    features[
                        f"{prefix}_mean_amplitude"
                    ] = np.nan
                    features[
                        f"{prefix}_band_power"
                    ] = np.nan
                    features[
                        f"{prefix}_bins_in_band"
                    ] = 0
        else:
            for fault_name, fundamental_hz in fault_frequencies.items():
                features[
                    f"{column}_{fault_name}_theoretical_hz"
                ] = fundamental_hz
                for harmonic in range(
                    1,
                    NUMBER_OF_HARMONICS + 1
                ):
                    target_hz = fundamental_hz * harmonic
                    prefix = (
                        f"{column}_{fault_name}_H{harmonic}"
                    )
                    measured = measure_frequency_band(
                        frequencies_all,
                        magnitude_all,
                        target_hz,
                        FREQUENCY_BANDWIDTH_HZ,
                    )
                    features[
                        f"{prefix}_target_hz"
                    ] = target_hz
                    features[
                        f"{prefix}_peak_amplitude"
                    ] = measured["peak_amplitude"]
                    features[
                        f"{prefix}_mean_amplitude"
                    ] = measured["mean_amplitude"]
                    features[
                        f"{prefix}_band_power"
                    ] = measured["band_power"]
                    features[
                        f"{prefix}_bins_in_band"
                    ] = measured["bins_in_band"]
        plot_signals.append(
            (column, frequencies_all, magnitude_all)
        )
    if plot_path is not None:
        plot_spectra(
            plot_signals,
            fault_frequencies,
            plot_path
        )
    return pd.DataFrame([features])

def plot_spectra(
    plot_signals,
    fault_frequencies,
    plot_path
):

    if not plot_signals:
        return

    n_channels = len(plot_signals)

    fig, axes = plt.subplots(
        n_channels,
        1,
        figsize=(12, max(4, 3.2 * n_channels)),
        squeeze=False
    )

    for ax, (
        channel,
        frequencies,
        magnitude
    ) in zip(axes[:, 0], plot_signals):

        ax.plot(
            frequencies[1:],
            magnitude[1:],
            linewidth=0.7
        )

        ax.set_title(
            f"{channel}: FFT magnitude spectrum"
        )

        ax.set_xlabel("Frequency (Hz)")

        ax.set_ylabel("FFT magnitude")

        ax.grid(True, alpha=0.3)

        if fault_frequencies is not None:

            for name, fundamental in fault_frequencies.items():

                for harmonic in range(
                    1,
                    NUMBER_OF_HARMONICS + 1
                ):

                    target = fundamental * harmonic

                    if target <= SAMPLING_FREQUENCY / 2:

                        ax.axvline(
                            target,
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.7,
                            label=(
                                f"{name} H{harmonic} "
                                f"({target:.1f} Hz)"
                            )
                        )

            ax.legend(
                fontsize=7,
                ncol=2
            )

    fig.tight_layout()

    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        plot_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


def process_dataset(dataset_name):

    folder = DATA_DIR / dataset_name

    if not folder.exists():

        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    files = sorted(
        file
        for file in folder.iterdir()
        if file.is_file()
    )

    if not files:

        raise FileNotFoundError(
            f"No recording files found in {folder}"
        )

    output_dir = PROCESSED_DIR / dataset_name

    plot_dir = output_dir / "plots" / "fft_spectra"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)

    print(f"FEATURE EXTRACTION: {dataset_name}")

    print("=" * 70)

    print("Total recordings:", len(files))

    all_rows = []

    fault_frequencies = calculate_bearing_frequencies(
        BEARING_PARAMETERS
    )

    if fault_frequencies is None:

        print(
            "WARNING: Bearing characteristic-frequency "
            "features are not configured."
        )

    else:

        print("\nTheoretical characteristic frequencies:")

        for name, value in fault_frequencies.items():
            print(f"  {name}: {value:.4f} Hz")
    for index, file_path in enumerate(files):

        dataset = load_recording(
            dataset_name,
            file_path.name
        )

        should_plot = (
            PLOT_EVERY_RECORDING or index == 0
        )

        plot_path = None

        if should_plot:

            plot_path = (
                plot_dir /
                f"{file_path.stem}_fft_spectra.png"
            )

        row = extract_features(
            dataset=dataset,
            recording_label=file_path.name,
            dataset_name=dataset_name,
            recording_index=index,
            plot_path=plot_path
        )
        all_rows.append(row)
        if (
            (index + 1) % 100 == 0
            or index + 1 == len(files)
        ):
            print(
                f"Processed {index + 1}/{len(files)} recordings"
            )
    results = pd.concat(
        all_rows,
        ignore_index=True
    )

    output_file = output_dir / "features_all.csv"
    results.to_csv(
        output_file,
        index=False,
        float_format="%.8g"
    )

    metadata = {

        **BEARING_PARAMETERS,

        "dataset": dataset_name,

        "sampling_frequency_hz": SAMPLING_FREQUENCY,

        "frequency_bandwidth_hz": FREQUENCY_BANDWIDTH_HZ,

        "number_of_harmonics": NUMBER_OF_HARMONICS,

        "fault_frequency_extraction_enabled": (
            fault_frequencies is not None
        ),

        "number_of_recordings": len(files),

        "number_of_channels": len(
            load_recording(
                dataset_name,
                files[0].name
            ).columns
        ),
    }

    pd.DataFrame([metadata]).to_csv(
        output_dir / "feature_extraction_parameters.csv",
        index=False
    )
    print("\nFEATURE EXTRACTION COMPLETED")
    print("Dataset:", dataset_name)
    print("Feature table shape:", results.shape)
    print("Saved features:", output_file)
    print("Saved plots:", plot_dir)
    return results

def main():
    for dataset_name in DATASETS:
        process_dataset(dataset_name)
    print("\n" + "=" * 70)
    print("ALL IMS DATASETS PROCESSED SUCCESSFULLY")
    print("=" * 70)
if __name__ == "__main__":
    main()