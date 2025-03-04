import os
import sys
import pandas as pd

DATA_DIR = "raw_data"
OUTPUT_DIR = "processed_data"


class Metadata:
    """
    fps: int            framerate of the data capture
    start_frame: int    starting frame from when data is "good" (motor is constant)
    id: int             identifier for the data run
    """

    def __init__(self, fps, start_frame, id):
        self.id = id
        self.start_frame = start_frame
        self.fps = fps

    def __repr__(self):
        # Zero-pad the id to two digits (e.g., id01_30)
        return f"id{self.id:02d}_{self.fps}"


def extract_metadata_from(file_name):
    """
    Extracts metadata from filename. Metadata schema:

    fps: int            framerate of the data capture
    start_frame: int    starting frame from when data is "good" (when drone starts flying)
    id: int             identifier for the data run
    """
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("-")

    if len(parts) != 3:
        raise ValueError(
            f"invalid metadata format: {file_name}. expected 3 parts, got {len(parts)}"
        )

    try:
        # Even if the file name has leading zeros (e.g., "001"), we convert to int.
        id = int(parts[0])
        start_frame = int(parts[1][1:])  # e.g., "f100" -> 100
        fps = parts[2]

        return Metadata(
            id=id,
            start_frame=start_frame,
            fps=fps,
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing filename '{file_name}': {str(e)}")


def extract_vicon_data(file_path, start_frame, duration):
    """
    Remove metadata/subheaders from CSV.
    Retains the following columns: Frame, RX, RY, RZ, TX, TY, TZ.
    Filters data only to "good" rows and skips rows that have no other values.
    Uses the start_frame from the vicon filename and extracts a window of 'duration' frames,
    where 'duration' is the sum of the pot input durations.
    """
    df = pd.read_csv(
        file_path,
        skiprows=[0, 1, 2, 4],  # these rows do not contain data. Row 4 is header
        usecols=[
            "Frame",
            "RX",
            "RY",
            "RZ",
            "TX",
            "TY",
            "TZ",
        ],
    )

    # Convert Frame column to numeric and drop rows where Frame is NaN.
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df.dropna(subset=["Frame"], inplace=True)

    # Drop rows where all motion data columns are missing.
    df.dropna(subset=["RX", "RY", "RZ", "TX", "TY", "TZ"], how="all", inplace=True)

    # Check if the vicon file covers the required window (based on pot duration).
    # We need frames from start_frame up to start_frame + duration.
    if df["Frame"].max() < start_frame + duration:
        print(
            "ERROR: the recorded vicon valid data window is less than the pot flight inputs!"
        )
        sys.exit(1)

    # Filter the DataFrame to extract the window based on the file name's start_frame and pot duration.
    df = df[(df["Frame"] >= start_frame) & (df["Frame"] < start_frame + duration)]

    # Reset Frame numbers to start from 1.
    df["Frame"] = range(1, len(df) + 1)

    return df


def extract_pot_data(file_path):
    """
    Reads the pot file, expands the pot data based on the duration of each row,
    and returns both the expanded pot DataFrame and the total duration (sum of durations).
    """
    df = pd.read_csv(
        file_path,
        skiprows=[],
        usecols=[
            "potZ",
            "potRot",
            "potY",
            "potX",
            "duration",
        ],
    )

    expanded_pot_data = []
    frame_list = []  # Track frame numbers

    frame_counter = 1  # Start from 1 instead of 0
    for _, row in df.iterrows():
        # Each row's pot data is repeated for the number of frames indicated by 'duration'
        expanded_pot_data.extend([row.drop("duration").values] * row["duration"])
        frame_list.extend(range(frame_counter, frame_counter + row["duration"]))
        frame_counter += row["duration"]

    expanded_pot_df = pd.DataFrame(
        expanded_pot_data, columns=["potZ", "potRot", "potY", "potX"]
    )

    # Assign frame numbers starting from 1.
    expanded_pot_df["Frame"] = frame_list

    # The total duration is the sum of the duration column.
    duration = df["duration"].sum()
    return expanded_pot_df, duration


def process_data(data_dir, output_dir):
    """
    Process input CSV files (vicon and pot), merge them, and write out a cleaned version.
    The window of frames extracted from the vicon data is determined dynamically based on
    the sum of the pot input durations, starting at the frame specified in the vicon file name.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vicon_files = [
        f for f in os.listdir(os.path.join(data_dir, "vicon")) if f.endswith(".csv")
    ]

    for vicon_file in vicon_files:
        metadata = extract_metadata_from(vicon_file)
        # Use zero-padded id for the pot file name (e.g., pot-01-30.csv)
        pot_file = f"pot-{metadata.id:02d}-{metadata.fps}.csv"

        vicon_file_path = os.path.join(data_dir, "vicon", vicon_file)
        pot_file_path = os.path.join(data_dir, "pot", pot_file)

        if not os.path.exists(pot_file_path):
            print(
                f"ERROR: pot file (expected {pot_file_path}) missing for {vicon_file}."
            )
            sys.exit(1)

        pot_df, duration = extract_pot_data(pot_file_path)
        vicon_df = extract_vicon_data(vicon_file_path, metadata.start_frame, duration)

        # Check that the extracted vicon frame count matches the pot duration.
        if len(vicon_df) != duration:
            print(
                f"Frame mismatch: Vicon has {len(vicon_df)} frames, but Pot expects {duration} frames for {vicon_file}. Skipping..."
            )
            continue

        # Merge the vicon and pot data on the 'Frame' column.
        df = pd.merge(vicon_df, pot_df, on="Frame", how="inner")

        if df.empty:
            print(f"No valid frames in {vicon_file}. Skipping...")
            continue

        print(df)

        out_name = f"{repr(metadata)}.csv"
        out_path = os.path.join(output_dir, out_name)

        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    process_data(DATA_DIR, OUTPUT_DIR)
