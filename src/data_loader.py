from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PROCESSED_DIR = DATA_DIR / "processed"

CHANNEL_NAMES = {

    "1st_test": [
        "Bx_1", "By_1",
        "Bx_2", "By_2",
        "Bx_3", "By_3",
        "Bx_4", "By_4"
    ],

    "2nd_test": [
        "B1", "B2", "B3", "B4"
    ],

    "3rd_test": [
        "B1", "B2", "B3", "B4"
    ]
}

# Number of VALID recordings per test (all downstream steps use only these).
# 1st test: all 2156 files are real measurements.
# 2nd test: 984 files, but the last two (2004.02.19.06.12.39 and .06.22.39)
#   read ~0.002 g on every channel, i.e. the sensors were already switched
#   off after the failure. They are not measurements -> 982 valid.
# 3rd test: 6324 files (the readme says 4448, but bearing 3 only degrades in
#   the last ~20 h, so the whole folder is used). The very last file reads
#   ~0.002 g on every channel (sensors off) -> 6323 valid.
VALID_RECORDINGS = {
    "1st_test": 2156,
    "2nd_test": 982,
    "3rd_test": 6323,
}

# A recording whose LOUDEST channel has an RMS below this value is treated as
# "sensors off". Real recordings never go below ~0.07 g, dead ones are ~0.002 g.
SENSOR_OFF_RMS = 0.02


def load_features(dataset_name):
    """
    Load features_all.csv (written by feature_extraction.py), keep only the
    valid recordings (VALID_RECORDINGS) and check that no sensor-off
    recording is left in the table.
    """

    input_file = PROCESSED_DIR / dataset_name / "features_all.csv"

    if not input_file.exists():
        raise FileNotFoundError(
            f"Feature table not found: {input_file}\n"
            "Run feature_extraction.py first."
        )

    features = pd.read_csv(input_file)

    n_valid = VALID_RECORDINGS[dataset_name]

    if len(features) < n_valid:
        raise ValueError(
            f"{dataset_name}: {len(features)} feature rows, "
            f"but {n_valid} valid recordings expected."
        )

    features = features.iloc[:n_valid].reset_index(drop=True)

    rms_columns = [c for c in features.columns if c.endswith("_rms")]
    loudest = features[rms_columns].max(axis=1)
    dead = features.loc[loudest < SENSOR_OFF_RMS, "recording"].tolist()

    if dead:
        raise ValueError(
            f"{dataset_name}: sensor-off recordings inside the valid range: "
            f"{dead}. Check VALID_RECORDINGS."
        )

    return features


def get_recording_files(dataset_name):

    if dataset_name not in CHANNEL_NAMES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    folder = DATA_DIR / dataset_name

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    # Fixed-width timestamp names: sorting by name = chronological order.
    # (iterdir() itself does not guarantee any order.)
    files = sorted(
        (f for f in folder.iterdir() if f.is_file()),
        key=lambda f: f.name
    )

    if not files:
        raise FileNotFoundError(
            f"No recordings found in: {folder}"
        )

    timestamps = []

    for file in files:

        try:
            timestamp = datetime.strptime(
                file.name,
                "%Y.%m.%d.%H.%M.%S"
            )

        except ValueError:
            raise ValueError(
                f"Unexpected recording filename: {file.name}"
            )

        timestamps.append(timestamp)

    if any(
        timestamps[i] > timestamps[i + 1]
        for i in range(len(timestamps) - 1)
    ):
        raise ValueError(
            f"Recordings in {dataset_name} are not chronological. "
            "Check their directory order before processing."
        )

    return files


def load_recording(dataset_name, filename):

    if dataset_name not in CHANNEL_NAMES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    file_path = DATA_DIR / dataset_name / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Recording not found: {file_path}"
        )

    dataset = pd.read_csv(
        file_path,
        sep="\t",
        header=None
    )

    expected_channels = CHANNEL_NAMES[dataset_name]

    if dataset.shape[1] != len(expected_channels):
        raise ValueError(
            f"{filename}: expected {len(expected_channels)} "
            f"channels, but found {dataset.shape[1]}."
        )

    dataset.columns = expected_channels

    dataset = dataset.apply(
        pd.to_numeric,
        errors="raise"
    )

    return dataset


def inspect_dataset(dataset_name):

    files = get_recording_files(dataset_name)

    first_file = files[0]

    dataset = load_recording(
        dataset_name,
        first_file.name
    )

    output_dir = (
        PROCESSED_DIR /
        dataset_name /
        "data_inspection"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = output_dir / "first_recording.csv"

    dataset.to_csv(
        csv_path,
        index=False
    )

    ax = dataset.plot(
        figsize=(14, 5),
        title=f"Bearing Vibration - {dataset_name}"
    )

    ax.set(
        xlabel="Sample index",
        ylabel="Vibration / acceleration (g)"
    )

    plt.tight_layout()

    plot_path = output_dir / "first_recording_vibration.png"

    plt.savefig(
        plot_path,
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    print("Total recordings:", len(files))
    print("First recording:", files[0].name)
    print("Last recording:", files[-1].name)

    print("Number of channels:", dataset.shape[1])
    print("Samples per channel:", dataset.shape[0])

    print("Channels:", dataset.columns.tolist())

    print("First recording saved:", csv_path)
    print("Vibration plot saved:", plot_path)


if __name__ == "__main__":

    for dataset_name in [
        "1st_test",
        "2nd_test",
        "3rd_test"
    ]:

        inspect_dataset(dataset_name)
