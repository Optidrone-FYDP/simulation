import pandas as pd
import sys
import matplotlib.pyplot as plt


def compute_velocity(df):
    return df.diff().fillna(0)


def compute_acceleration(df):
    return df.diff().diff().fillna(0)


def compare_csvs(lstm_file, actual_file, plot_type):
    # For translation we compare TX, TY, TZ; for rotation, RX, RY, RZ.
    translation_columns = ["TX", "TY", "TZ"]
    rotation_columns = ["RX", "RY", "RZ"]

    if plot_type == "rotation":
        columns = rotation_columns
        title = "Rotation: LSTM vs Actual"
    else:
        columns = translation_columns
        if plot_type == "velocity":
            title = "Velocity: LSTM vs Actual"
        elif plot_type == "acceleration":
            title = "Acceleration: LSTM vs Actual"
        else:
            title = "Position: LSTM vs Actual"

    lstm_df = pd.read_csv(lstm_file, usecols=columns)
    actual_df = pd.read_csv(actual_file, usecols=columns)

    if plot_type == "velocity":
        lstm_df = compute_velocity(lstm_df)
        actual_df = compute_velocity(actual_df)
    elif plot_type == "acceleration":
        lstm_df = compute_acceleration(lstm_df)
        actual_df = compute_acceleration(actual_df)

    plt.figure(figsize=(12, 6))
    for col in columns:
        plt.plot(lstm_df[col], label=f"LSTM {col}", linestyle=":")
        plt.plot(actual_df[col], label=f"Actual {col}", linestyle="-")

    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.savefig("temppppp.png")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_csvs.py <lstm_csv> <actual_csv> <plot_type>")
        print("plot_type must be one of: position, velocity, acceleration, rotation")
        sys.exit(1)

    compare_csvs(sys.argv[1], sys.argv[2], sys.argv[3])
