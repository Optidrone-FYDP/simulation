import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

<<<<<<< Updated upstream
MODEL_PATH = "drone_movement_model_5k.pt"
SEQ_LENGTH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneMovementModel, self).__init__()
=======
class DroneNARXModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneNARXModel, self).__init__()
>>>>>>> Stashed changes
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   

MODEL_PATH = "drone_movement_model_5k.pt"
SEQ_LENGTH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def expand_pot_inputs(df):
    sequence = []
    for _, row in df.iterrows():
        pot = [row["potX"], row["potY"], row["potZ"]]
        duration = int(row["duration"])
        sequence.extend([pot] * duration)
    return np.array(sequence, dtype=np.float32)


if len(sys.argv) != 8:
    print(
        f"Usage: python {sys.argv[0]} <simulation_inputs.csv> <start_RX> <start_RY> <start_RZ> <start_TX> <start_TY> <start_TZ>"
    )
    sys.exit(1)

input_file = sys.argv[1]
start_RX, start_RY, start_RZ, start_TX, start_TY, start_TZ = map(float, sys.argv[2:8])

df_input = pd.read_csv(input_file)
expanded_sequence = expand_pot_inputs(df_input)
total_frames = len(expanded_sequence)

model = DroneNARXModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

<<<<<<< Updated upstream
current_position = np.array(
    [start_RX, start_RY, start_RZ, start_TX, start_TY, start_TZ], dtype=np.float32
)
=======
input_scaler = joblib.load("input_scaler.pkl")
output_scaler = joblib.load("output_scaler.pkl")

current_position = np.array([start_RX, start_RY, start_RZ, start_TX, start_TY, start_TZ], dtype=np.float32)
>>>>>>> Stashed changes
positions = [current_position.copy()]

initial_sequence = []
for i in range(SEQ_LENGTH):
    exogeneous_input = expanded_sequence[i]
    combined = np.concatenate([exogeneous_input, current_position])
    initial_sequence.append(combined)
initial_sequence = np.array(initial_sequence, dtype=np.float32)

initial_sequence_scaled = input_scaler.transform(initial_sequence)
input_seq = torch.tensor(initial_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

for i in range(total_frames - SEQ_LENGTH):
<<<<<<< Updated upstream
    input_seq = (
        torch.tensor(expanded_sequence[i : i + SEQ_LENGTH], dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )
    with torch.no_grad():
        pred = model(input_seq).cpu().numpy().flatten()
    current_position = pred
=======
    with torch.no_grad():
        pred_scaled = model(input_seq)
    pred = output_scaler.inverse_transform(pred_scaled.cpu().numpy()).flatten()

    current_position = pred
    input_seq = torch.tensor(expanded_sequence[i:i+SEQ_LENGTH], dtype=torch.float32).unsqueeze(0).to(DEVICE)
>>>>>>> Stashed changes
    positions.append(current_position.copy())

    exogenous_input = expanded_sequence[i]
    combined = np.concatenate([exogenous_input, current_position])
    combined_scaled = input_scaler.transform(combined.reshape(1, -1)).flatten()
    input_seqeucence_np = input_seq.cpu().numpy()
    input_seqeucence_np = np.concatenate([input_seqeucence_np[:, 1:, :], combined_scaled.reshape(1, 1, -1)], axis=1)

    input_seq = torch.tensor(input_seqeucence_np, dtype=torch.float32).to(DEVICE)

labels = ["RX", "RY", "RZ", "TX", "TY", "TZ"]
output_df = pd.DataFrame(positions, columns=labels)
output_df["Frame"] = range(1, len(positions) + 1)
output_df = output_df[["Frame"] + labels]

output_df.to_csv("predictions.csv", index=False)

output_df[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]] = (
    output_df[labels].diff().fillna(0)
)
output_df[["A_RX", "A_RY", "A_RZ", "A_TX", "A_TY", "A_TZ"]] = (
    output_df[["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]].diff().fillna(0)
)

fig, axs = plt.subplots(3, 1, figsize=(15, 18))

for label in labels:
    axs[0].plot(output_df["Frame"], output_df[label], label=label)
axs[0].set_title("Position Over Time")
axs[0].set_xlabel("Frame")
axs[0].set_ylabel("Position")
axs[0].legend()
axs[0].grid(True)

for label in ["V_RX", "V_RY", "V_RZ", "V_TX", "V_TY", "V_TZ"]:
    axs[1].plot(output_df["Frame"], output_df[label], label=label)
axs[1].set_title("Velocity Over Time")
axs[1].set_xlabel("Frame")
axs[1].set_ylabel("Velocity")
axs[1].legend()
axs[1].grid(True)

for label in ["A_RX", "A_RY", "A_RZ", "A_TX", "A_TY", "A_TZ"]:
    axs[2].plot(output_df["Frame"], output_df[label], label=label)
axs[2].set_title("Acceleration Over Time")
axs[2].set_xlabel("Frame")
axs[2].set_ylabel("Acceleration")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("predictions.png")
