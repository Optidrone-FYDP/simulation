import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "drone_movement_model_lstm_0.2.pt"
SEQ_LENGTH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROT_NORM_FACTOR = 2 * np.pi


class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def expand_pot_inputs(df):
    sequence = []
    for _, row in df.iterrows():
        pot = [row["potX"], row["potY"], row["potZ"], row["potRot"]]
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
current_position = np.array(
    [start_TX, start_TY, start_TZ, start_RX, start_RY, start_RZ], dtype=np.float32
)

df_input = pd.read_csv(input_file)
expanded_sequence = expand_pot_inputs(df_input)
total_frames = len(expanded_sequence)

model = DroneMovementModel().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

positions = [current_position.copy()]

for i in range(total_frames - SEQ_LENGTH):
    input_seq = (
        torch.tensor(expanded_sequence[i : i + SEQ_LENGTH], dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )
    with torch.no_grad():
        delta = model(input_seq).cpu().numpy().flatten()
    delta[3:] = delta[3:] * ROT_NORM_FACTOR
    current_position = current_position + delta
    positions.append(current_position.copy())

labels = ["TX", "TY", "TZ", "RX", "RY", "RZ"]
output_df = pd.DataFrame(positions, columns=labels)
output_df["Frame"] = range(1, len(positions) + 1)
output_df = output_df[["Frame"] + labels]

output_df.to_csv("predictions.csv", index=False)

output_df[["V_TX", "V_TY", "V_TZ", "V_RX", "V_RY", "V_RZ"]] = (
    output_df[labels].diff().fillna(0)
)
output_df[["A_TX", "A_TY", "A_TZ", "A_RX", "A_RY", "A_RZ"]] = (
    output_df[["V_TX", "V_TY", "V_TZ", "V_RX", "V_RY", "V_RZ"]].diff().fillna(0)
)

fig, axs = plt.subplots(3, 1, figsize=(15, 18))

for label in labels:
    axs[0].plot(output_df["Frame"], output_df[label], label=label)
axs[0].set_title("Position Over Time")
axs[0].set_xlabel("Frame")
axs[0].set_ylabel("Position")
axs[0].legend()
axs[0].grid(True)

for label in ["V_TX", "V_TY", "V_TZ", "V_RX", "V_RY", "V_RZ"]:
    axs[1].plot(output_df["Frame"], output_df[label], label=label)
axs[1].set_title("Velocity Over Time")
axs[1].set_xlabel("Frame")
axs[1].set_ylabel("Velocity")
axs[1].legend()
axs[1].grid(True)

for label in ["A_TX", "A_TY", "A_TZ", "A_RX", "A_RY", "A_RZ"]:
    axs[2].plot(output_df["Frame"], output_df[label], label=label)
axs[2].set_title("Acceleration Over Time")
axs[2].set_xlabel("Frame")
axs[2].set_ylabel("Acceleration")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("predictions.png")
