import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_DIR, OUTPUT_DIR, BATCH_SIZE, INTERPOLATE, CUTOFF


class Metadata:
    """
    fps: int            framerate of the data capture
    start_frame: int    starting frame from when data is "good" (motor is constant)
    end_frame: int      end frame of "good" data
    pot_x: int          potentiometer value of x (0-63 is ?) (64-128 is ?)
    pot_y: int          potentiometer value of y (0-63 is down) (64-128 is up)
    pot_z: int          potentiometer value of z (0-63 is ?) (64-128 is ?)
    id: int             identifier for the data run
    """

    def __init__(self, fps, start_frame, end_frame, pot_x, pot_y, pot_z, id):
        self.id = id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.pot_x = pot_x
        self.pot_y = pot_y
        self.pot_z = pot_z
        self.fps = fps

    def __repr__(self):
        return f"id{self.id}_x{self.pot_x}_y{self.pot_y}_z{self.pot_z}_fps{self.fps}"


def extract_metadata_from(file_name):
    inputs = file_name.split(".")[0].split("-")

    if len(inputs) != 7:
        raise ValueError(
            f"Invalid metadata format: {file_name}. Expected 7 parts, got {len(inputs)}"
        )

    # TODO: more validation. Also, consider making the naming format easily ordered
    try:
        return Metadata(
            id=int(inputs[0]),
            start_frame=int(inputs[1][1:]),
            end_frame=int(inputs[2][1:]),
            pot_x=int(inputs[3][1:]),
            pot_y=int(inputs[4][1:]),
            pot_z=int(inputs[5][1:]),
            fps=int(inputs[6]),
        )
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid metadata format: {file_name}. Error: {str(e)}")


def extract_position_data_from(
    file_path, start_frame=None, end_frame=None, interpolate=False
):
    df = pd.read_csv(
        file_path, skiprows=[0, 1, 2, 4], usecols=["Frame", "TX", "TY", "TZ"]
    )
    df.set_index("Frame", inplace=True)

    if interpolate:
        df.interpolate(
            method="linear", inplace=True
        )  # interpolate missing values-- csv sometimes has blank frames

    if start_frame is not None and end_frame is not None and CUTOFF:
        df = df[start_frame:end_frame]

    positions = df[["TX", "TY", "TZ"]].astype(float)
    return positions.values


class FlightRun:
    def __init__(self, metadata, positions):
        self.metadata = metadata
        self.positions = positions

    def __repr__(self):
        return f"FlightRun({self.metadata})"


def read_csv_files(data_dir, batch_size=100, interpolate=False):
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    data = []

    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        batch_data = []

        for file_name in batch_files:
            file_path = os.path.join(data_dir, file_name)

            metadata = extract_metadata_from(file_name)
            positions = extract_position_data_from(
                file_path, metadata.start_frame, metadata.end_frame, interpolate
            )

            print(FlightRun(metadata, positions))

            batch_data.append(FlightRun(metadata, positions))

        data.extend(batch_data)

    return data


def calculate_acceleration(positions, fps):
    time_step = 1.0 / fps
    velocities = np.diff(positions, axis=0) / time_step
    accelerations = np.diff(velocities, axis=0) / time_step
    return accelerations, velocities


def plot_results(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_mapping = {}

    for trial in data:
        print(trial)
        positions = trial.positions
        print(positions)
        metadata = trial.metadata

        accelerations, velocities = calculate_acceleration(positions, metadata.fps)
        avg_acceleration = np.mean(accelerations, axis=0)

        pot_key = f"{metadata.pot_x:03d}{metadata.pot_y:03d}{metadata.pot_z:03d}"
        result_mapping[pot_key] = [
            avg_acceleration[0],
            avg_acceleration[1],
            avg_acceleration[2],
        ]

        plt.figure(figsize=(12, 12))

        # Plot positions
        plt.subplot(3, 1, 1)
        plt.plot(positions[:, 0], label="Position X")
        plt.plot(positions[:, 1], label="Position Y")
        plt.plot(positions[:, 2], label="Position Z")
        plt.title(
            f"Positions for Potentiometers: X={metadata.pot_x}, Y={metadata.pot_y}, Z={metadata.pot_z}"
        )
        plt.xlabel("Frame")
        plt.ylabel("Position (mm)")
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(accelerations[:, 0], label="Acceleration X")
        plt.plot(accelerations[:, 1], label="Acceleration Y")
        plt.plot(accelerations[:, 2], label="Acceleration Z")
        plt.axhline(
            avg_acceleration[0], color="b", linestyle="--", label="Avg Acceleration X"
        )
        plt.axhline(
            avg_acceleration[1],
            color="#ffa500",
            linestyle="--",
            label="Avg Acceleration Y",
        )
        plt.axhline(
            avg_acceleration[2], color="g", linestyle="--", label="Avg Acceleration Z"
        )
        plt.title("Calculated Accelerations")
        plt.xlabel("Frame")
        plt.ylabel("Acceleration (mm/s^2)")
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(velocities[:, 0], label="Velocity X")
        plt.plot(velocities[:, 1], label="Velocity Y")
        plt.plot(velocities[:, 2], label="Velocity Z")
        plt.title("Calculated Velocities")
        plt.xlabel("Frame")
        plt.ylabel("Velocity (mm/s)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{repr(metadata)}.png"))
        plt.close()

    with open(f"{output_dir}/accelerations.json", "w") as json_file:
        json.dump(result_mapping, json_file, indent=4)

    pot_values = []
    avg_acc_y = []

    for key, value in result_mapping.items():
        pot_y = int(key[3:6])  # Extract Y potentiometer value
        acc_y = float(value[1])  # Extract average Y acceleration
        pot_values.append(pot_y)
        avg_acc_y.append(acc_y)

    plt.figure(figsize=(10, 6))
    plt.scatter(pot_values, avg_acc_y, c="blue", marker="o")

    fit = np.polyfit(pot_values, avg_acc_y, 1)
    fit_fn = np.poly1d(fit)

    line_equation = f"y = {fit[0]:.2f}x + {fit[1]:.2f}"
    plt.text(
        0.05,
        0.95,
        line_equation,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
    )

    # Plot line of best fit
    plt.plot(pot_values, fit_fn(pot_values), "r--", label="Best fit line")
    plt.axvline(x=64, color="green", linestyle="--", label="pot_y = 64")

    plt.title("Potentiometer Y Values vs. Average Y Acceleration")
    plt.xlabel("Potentiometer Y Value")
    plt.ylabel("Average Y Acceleration (mm/s^2)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "potentiometer_y_vs_acceleration_y.png"))
    plt.close()


# Main function
def main():
    data = read_csv_files(DATA_DIR, batch_size=BATCH_SIZE, interpolate=INTERPOLATE)
    plot_results(data, OUTPUT_DIR)


if __name__ == "__main__":
    main()
