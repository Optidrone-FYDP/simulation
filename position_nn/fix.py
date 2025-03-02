import pandas as pd
import glob
import os

# Define the desired column order
desired_order = [
    "Frame",
    "RX",
    "RY",
    "RZ",
    "TX",
    "TY",
    "TZ",
    "potZ",
    "potRot",
    "potY",
    "potX",
]

# Pattern to match all CSV files in the current directory (modify the path as needed)
csv_files = glob.glob("*.csv")

for file in csv_files:
    df = pd.read_csv(file)

    # Add 'potRot' with a constant value of 64 if it doesn't exist
    if "potRot" not in df.columns:
        df["potRot"] = 64

    # If all desired columns exist, reorder them
    if all(col in df.columns for col in desired_order):
        df = df[desired_order]

    # Overwrite the original file (or change the path to save to a different location)
    df.to_csv(file, index=False)
    print(f"Modified {os.path.basename(file)}")
