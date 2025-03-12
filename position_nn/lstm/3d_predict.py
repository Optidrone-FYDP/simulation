#!/usr/bin/env python
import csv
import argparse
import os, sys, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set a seed for reproducibility.
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

######################################
# 1. Compress Raw Test Data
######################################
def pot_values_match(val1, val2, tol):
    return all(abs(a - b) <= tol for a, b in zip(val1, val2))

def compress_pot_values(input_filename, output_filename, tolerance):
    groups = []
    prev_vals = None
    count = 0
    with open(input_filename, newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                potX = int(round(float(row["potX"])))
                potY = int(round(float(row["potY"])))
                potZ = int(round(float(row["potZ"])))
            except KeyError:
                raise ValueError("Input CSV must have columns: potX, potY, potZ")
            current_vals = (potX, potY, potZ)
            if prev_vals is None:
                prev_vals = current_vals
                count = 1
            elif pot_values_match(current_vals, prev_vals, tolerance):
                count += 1
            else:
                # Although we output potRot, it's unused downstream.
                groups.append(prev_vals + (64, count))
                prev_vals = current_vals
                count = 1
        if prev_vals is not None:
            groups.append(prev_vals + (64, count))
    with open(output_filename, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["potX", "potY", "potZ", "potRot", "duration"])
        writer.writerows(groups)
    print(f"Compressed simulation input saved to '{output_filename}'.")

######################################
# 2. Prediction from Simulation CSV
######################################
# Translation-only model: 3 inputs (potX, potY, potZ)
class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=2):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def normalize_pot(x):
    return (x - 64.0) / 64.0

def expand_pot_inputs(df):
    sequence = []
    for _, row in df.iterrows():
        pot = [normalize_pot(row["potX"]),
               normalize_pot(row["potY"]),
               normalize_pot(row["potZ"])]
        duration = int(row["duration"])
        sequence.extend([pot] * duration)
    return np.array(sequence, dtype=np.float32)

def run_prediction(simulation_file, model_path, start_tx, start_ty, start_tz,
                   seq_length=20, device="cpu", output_filename="predictions.csv"):
    df_sim = pd.read_csv(simulation_file)
    expanded_sequence = expand_pot_inputs(df_sim)
    total_frames = len(expanded_sequence)
    current_position = np.array([start_tx, start_ty, start_tz], dtype=np.float32)
    
    model = DroneMovementModel(input_dim=3, hidden_dim=64, output_dim=3, num_layers=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    states = [{"TX": current_position[0], "TY": current_position[1], "TZ": current_position[2]}]
    for i in range(total_frames - seq_length):
        input_seq = torch.tensor(expanded_sequence[i : i + seq_length], dtype=torch.float32)\
                        .unsqueeze(0).to(device)
        with torch.no_grad():
            delta = model(input_seq).cpu().numpy().flatten()
        current_position += delta
        states.append({"TX": current_position[0],
                       "TY": current_position[1],
                       "TZ": current_position[2]})
    
    pred_df = pd.DataFrame(states)
    pred_df["Frame"] = range(1, len(states) + 1)
    pred_df = pred_df[["Frame", "TX", "TY", "TZ"]]
    pred_df.to_csv(output_filename, index=False)
    print(f"Predictions saved to {output_filename}.")
    return output_filename

######################################
# 3. 3D Comparison Plot
######################################
def generate_3d_comparison(prediction_file, raw_file, output_filename="comparison_3d.png"):
    pred_df = pd.read_csv(prediction_file)
    actual_df = pd.read_csv(raw_file)
    
    # Ensure required columns exist.
    for col in ["TX", "TY", "TZ"]:
        if col not in pred_df.columns:
            print(f"Column {col} not found in prediction CSV.")
            return
        if col not in actual_df.columns:
            print(f"Column {col} not found in raw input CSV.")
            return

    # Align trajectories to start at (0,0,0)
    for col in ["TX", "TY", "TZ"]:
        pred_df[col] = pred_df[col] - pred_df[col].iloc[0]
        actual_df[col] = actual_df[col] - actual_df[col].iloc[0]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pred_df["TX"], pred_df["TY"], pred_df["TZ"],
            label="Predicted Trajectory", marker="o", markersize=4, linestyle="-", linewidth=1.5)
    ax.plot(actual_df["TX"], actual_df["TY"], actual_df["TZ"],
            label="Actual Trajectory", marker="^", markersize=4, linestyle="--", linewidth=1.5)
    ax.set_xlabel("TX", fontsize=12)
    ax.set_ylabel("TY", fontsize=12)
    ax.set_zlabel("TZ", fontsize=12)
    ax.set_title("3D Flight Path Comparison (Aligned to 0,0,0)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    # Set equal aspect ratio for all axes (approximate)
    max_range = np.array([pred_df["TX"].max()-pred_df["TX"].min(), 
                          pred_df["TY"].max()-pred_df["TY"].min(), 
                          pred_df["TZ"].max()-pred_df["TZ"].min()]).max() / 2.0
    mid_x = (pred_df["TX"].max()+pred_df["TX"].min()) * 0.5
    mid_y = (pred_df["TY"].max()+pred_df["TY"].min()) * 0.5
    mid_z = (pred_df["TZ"].max()+pred_df["TZ"].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"3D comparison plot saved as {output_filename}.")
    plt.show()

######################################
# Main: Combined Execution
######################################
def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: compress raw test data, run prediction, and generate a 3D comparison graph."
    )
    parser.add_argument("--raw", required=True,
                        help="Path to the raw test CSV file (headers: Frame,RX,RY,RZ,TX,TY,TZ,potZ,potRot,potY,potX)")
    parser.add_argument("--model", required=True,
                        help="Path to the trained model file")
    parser.add_argument("--start", nargs=3, type=float, required=True, metavar=("START_TX", "START_TY", "START_TZ"),
                        help="Starting translation values (TX, TY, TZ)")
    parser.add_argument("--tolerance", type=int, default=0,
                        help="Tolerance for grouping pot values (default: 0)")
    parser.add_argument("--sim_out", default="simulation.csv",
                        help="Output filename for compressed simulation CSV")
    parser.add_argument("--pred_out", default="predictions.csv",
                        help="Output filename for predictions CSV")
    parser.add_argument("--comp_out", default="comparison_3d.png",
                        help="Output filename for 3D comparison plot")
    args = parser.parse_args()

    # Step 1: Compress raw test data to simulation CSV.
    compress_pot_values(args.raw, args.sim_out, args.tolerance)
    # Step 2: Run prediction using the simulation CSV and the trained model.
    run_prediction(args.sim_out, args.model, args.start[0], args.start[1], args.start[2],
                   seq_length=20, device=DEVICE, output_filename=args.pred_out)
    # Step 3: Generate a 3D comparison plot.
    generate_3d_comparison(args.pred_out, args.raw, output_filename=args.comp_out)

if __name__ == "__main__":
    main()
