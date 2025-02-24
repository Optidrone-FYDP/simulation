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
        return f"id{self.id}_{self.fps}"


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
        id = int(parts[0])  # 001
        start_frame = int(parts[1][1:])  # f100 -> 100
        fps = (parts[2])  # 30

        return Metadata(
            id=id,
            start_frame=start_frame,
            fps=fps,
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing filename '{file_name}': {str(e)}")


def extract_vicon_data(file_path, start_frame, duration):
    """
    Remove metadata/subheaders from CSV
    Retains the following columns: Frame, RX, RY, RZ, TX, TY, TZ, potX, potY, potZ
    Filter data only to good rows
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

    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")

    df.dropna(subset=["Frame"], inplace=True)

    if df.shape[0] < start_frame + duration:
        print(f"ERROR: the recorded vicon valid data window is less than pot flight inputs!")
        sys.exit(1)
    df = df[(df["Frame"] >= start_frame) & (df["Frame"] <= start_frame + duration)]
    df.set_index("Frame", inplace=True)
    return df


def extract_pot_data(file_path):
    df = pd.read_csv(
        file_path,
        skiprows=[],
        usecols=[
            "timestamp",
            "potZ",
            "potRot",
            "potY",
            "potX",
        ],
    )

    duration = df.shape[0]
    return df, duration

def process_data(data_dir, output_dir):
    """
    Process input CSV and write out a cleaned version.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vicon_files = [
        f for f in os.listdir(os.path.join(data_dir, "vicon")) if f.endswith(".csv")
    ]

    for vicon_file in vicon_files:
        metadata = extract_metadata_from(vicon_file)
        pot_file = f"pot-{metadata.id}-{metadata.fps}.csv"

        vicon_file_path = os.path.join(data_dir, "vicon", vicon_file)
        pot_file_path = os.path.join(data_dir, "pot", pot_file)

        if not os.path.exists(pot_file_path):
            print(f"ERROR: pot file (expected {pot_file_path}) missing for {vicon_file}.")
            sys.exit(1)

        pot_df, duration = extract_pot_data(pot_file_path,)
        vicon_df = extract_vicon_data(vicon_file_path, metadata.start_frame, duration)

        if len(vicon_df) != len(pot_df):
            print(
                f"Row mismatch between vicon ({len(vicon_df)}) and pot ({len(pot_df)}) data for {vicon_file}. Skipping..."
            )
            continue

        # align the indices to avoid mismatches
        pot_df = pot_df.reset_index(drop=True)
        vicon_df = vicon_df.reset_index(drop=True)

        # column-wise concatenation
        df = pd.concat([vicon_df, pot_df], axis=1)

        # Reset the index to write a clean output
        df.reset_index(drop=True, inplace=True)

        # Merge the two DataFrames
        # df = pd.concat([df1, df2], axis=1)
        df = pd.concat([vicon_df, pot_df.reset_index(drop=True)], axis=1)

        # Reset the index to write a clean output
        df.reset_index(inplace=True)

        if df.empty:
            print(f"No valid frames in {vicon_file}. Skipping...")
            continue

        print(df)

        out_name = f"{repr(metadata)}.csv"
        out_path = os.path.join(output_dir, out_name)

        # save CSV
        df.reset_index(inplace=True)
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    process_data(DATA_DIR, OUTPUT_DIR)
