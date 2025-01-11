import os
import pandas as pd

DATA_DIR = "raw_data"
OUTPUT_DIR = "processed_data"


class Metadata:
    """
    fps: int            framerate of the data capture
    start_frame: int    starting frame from when data is "good" (motor is constant)
    end_frame: int      end frame of "good" data
    id: int             identifier for the data run
    """

    def __init__(self, fps, start_frame, end_frame, id):
        self.id = id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.fps = fps

    def __repr__(self):
        return f"id{self.id}_{self.fps}"


def extract_metadata_from(file_name):
    """
    Extracts metadata from filename. Metadata schema:

    fps: int            framerate of the data capture
    start_frame: int    starting frame from when data is "good" (motor is constant)
    end_frame: int      end frame of "good" data
    id: int             identifier for the data run
    """
    base_name = os.path.splitext(file_name)[0]
    parts = base_name.split("-")

    if len(parts) != 4:
        raise ValueError(
            f"invalid metadata format: {file_name}. expected 4 parts, got {len(parts)}"
        )

    try:
        run_id_str = parts[0]  # "01"
        start_frame_str = parts[1]  # "f100"
        end_frame_str = parts[2]  # "f460"
        # pot_x_str = parts[3]       # "x064"
        # pot_y_str = parts[4]       # "y128"
        # pot_z_str = parts[5]       # "z064"
        fps_str = parts[3]  # "30"

        return Metadata(
            id=int(run_id_str),
            start_frame=int(start_frame_str[1:]),
            end_frame=int(end_frame_str[1:]),
            # pot_x=int(pot_x_str[1:]),
            # pot_y=int(pot_y_str[1:]),
            # pot_z=int(pot_z_str[1:]),
            fps=int(fps_str),
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"Error parsing filename '{file_name}': {str(e)}")


def extract_data(file_path, start_frame, end_frame):
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

    # slice by start to end frame, if exists
    df = df[(df["Frame"] >= start_frame) & (df["Frame"] <= end_frame)]
    df.set_index("Frame", inplace=True)
    return df


def extract_pot_data(file_path, start_frame, end_frame):
    """
    Remove metadata/subheaders from CSV
    Retains the following columns: Frame, RX, RY, RZ, TX, TY, TZ, potX, potY, potZ
    Filter data only to good rows
    """
    df = pd.read_csv(
        file_path,
        skiprows=[],
        usecols=[
            "timestamp",
            "up/down pot",
            "rotation pot",
            "front/back pot",
            "left/right pot",
        ],
    )

    # this shouldnt even be needed
    df = df.head(end_frame - start_frame + 1)
    return df


def process_data(data_dir, output_dir):
    """
    Process input CSV and write out a cleaned version.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all CSV files in the vicon directory
    vicon_files = [
        f for f in os.listdir(os.path.join(data_dir, "vicon")) if f.endswith(".csv")
    ]

    for vicon_file in vicon_files:
        # Extract metadata from the vicon file name
        metadata = extract_metadata_from(vicon_file)  # Custom function for metadata

        # Construct the corresponding pot file name
        pot_file = f"pot-{vicon_file}"

        # Paths to vicon and pot files
        vicon_file_path = os.path.join(data_dir, "vicon", vicon_file)
        pot_file_path = os.path.join(data_dir, "pot", pot_file)

        # Check if corresponding pot file exists
        if not os.path.exists(pot_file_path):
            print(f"Pot file missing for {vicon_file}. Skipping...")
            continue

        # Read and filter the vicon CSV
        df1 = extract_data(
            vicon_file_path,
            start_frame=metadata.start_frame,
            end_frame=metadata.end_frame,
        )

        # Read and filter the pot CSV
        df2 = extract_pot_data(
            pot_file_path,
            start_frame=metadata.start_frame,
            end_frame=metadata.end_frame,
        )

        # Merge the two DataFrames side by side
        if len(df1) != len(df2):
            print(
                f"Row mismatch between vicon ({len(df1)}) and pot ({len(df2)}) data for {vicon_file}. Skipping..."
            )
            continue

        # Align the indices to avoid mismatches
        df2 = df2.reset_index(drop=True)
        df1 = df1.reset_index(drop=True)

        # Concatenate the dataframes column-wise
        df = pd.concat([df1, df2], axis=1)

        # Reset the index to write a clean output
        df.reset_index(drop=True, inplace=True)

        # Merge the two DataFrames
        # df = pd.concat([df1, df2], axis=1)
        df = pd.concat([df1, df2.reset_index(drop=True)], axis=1)

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
