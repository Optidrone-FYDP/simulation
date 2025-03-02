import pandas as pd
import sys
import matplotlib.pyplot as plt


def compute_velocity(df):
    return df.diff().fillna(0)


def compute_acceleration(df):
    return df.diff().diff().fillna(0)


def compare_csvs(narx_file, lstm_file, actual_file, plot_type):
    all_columns = ["RX", "RY", "RZ", "TX", "TY", "TZ"]
    translation_columns = ["TX", "TY", "TZ"]
    rotation_columns = ["RX", "RY", "RZ"]

    if plot_type == "rotation":
        columns = rotation_columns
        title = "Rotation: NARX vs LSTM vs Actual"
    else:
        columns = translation_columns
        title = (
            "Velocity: NARX vs LSTM vs Actual"
            if plot_type == "velocity"
            else (
                "Acceleration: NARX vs LSTM vs Actual"
                if plot_type == "acceleration"
                else "Position: NARX vs LSTM vs Actual"
            )
        )

    narx_df = pd.read_csv(narx_file, usecols=columns)
    lstm_df = pd.read_csv(lstm_file, usecols=columns)
    actual_df = pd.read_csv(actual_file, usecols=columns)

    if plot_type == "velocity":
        narx_df, lstm_df, actual_df = (
            compute_velocity(narx_df),
            compute_velocity(lstm_df),
            compute_velocity(actual_df),
        )
    elif plot_type == "acceleration":
        narx_df, lstm_df, actual_df = (
            compute_acceleration(narx_df),
            compute_acceleration(lstm_df),
            compute_acceleration(actual_df),
        )

    plt.figure(figsize=(12, 6))
    for col in columns:
        plt.plot(narx_df[col], label=f"NARX {col}", linestyle="--")
        plt.plot(lstm_df[col], label=f"LSTM {col}", linestyle=":")
        plt.plot(actual_df[col], label=f"Actual {col}", linestyle="-")

    plt.title(title)
    plt.legend()
    plt.xlabel("Frame Index")
    plt.ylabel("Values")
    plt.grid()
    plt.show()
    plt.savefig("temppp.png")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: python compare_csvs.py <narx_csv> <lstm_csv> <actual_csv> <plot_type>"
        )
        print("plot_type must be one of: position, velocity, acceleration, rotation")
        sys.exit(1)

    compare_csvs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
