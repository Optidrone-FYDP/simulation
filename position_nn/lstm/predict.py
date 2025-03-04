import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Toggle to predict rotation along with translation.
PREDICT_ROTATION = False  # Set to False to predict only translation

MODEL_PATH = "models/drone_movement_model_lstm_0.4.pt"
SEQ_LENGTH = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROT_NORM_FACTOR = 2 * np.pi

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

def normalize_pot(x):
    """Normalize potentiometer value: from [0, 128] (with 64 as neutral) to [-1, 1]."""
    return (x - 64.0) / 64.0

class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3, num_layers=2):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def expand_pot_inputs(df):
    sequence = []
    for _, row in df.iterrows():
        pot = [
            normalize_pot(row["potX"]),
            normalize_pot(row["potY"]),
            normalize_pot(row["potZ"]),
            normalize_pot(row["potRot"]),
        ]
        duration = int(row["duration"])
        sequence.extend([pot] * duration)
    return np.array(sequence, dtype=np.float32)

if PREDICT_ROTATION:
    if len(sys.argv) != 8:
        print(f"Usage: python {sys.argv[0]} <simulation_inputs.csv> <start_TX> <start_TY> <start_TZ> <start_Rx> <start_Ry> <start_Rz>")
        sys.exit(1)
    input_file = sys.argv[1]
    start_TX, start_TY, start_TZ = map(float, sys.argv[2:5])
    start_Rx, start_Ry, start_Rz = map(float, sys.argv[5:8])
    current_position = np.array([start_TX, start_TY, start_TZ], dtype=np.float32)
    current_rotation = np.array([start_Rx, start_Ry, start_Rz], dtype=np.float32)
else:
    if len(sys.argv) != 5:
        print(f"Usage: python {sys.argv[0]} <simulation_inputs.csv> <start_TX> <start_TY> <start_TZ>")
        sys.exit(1)
    input_file = sys.argv[1]
    start_TX, start_TY, start_TZ = map(float, sys.argv[2:5])
    current_position = np.array([start_TX, start_TY, start_TZ], dtype=np.float32)

df_input = pd.read_csv(input_file)
expanded_sequence = expand_pot_inputs(df_input)
total_frames = len(expanded_sequence)

output_dim = 6 if PREDICT_ROTATION else 3
model = DroneMovementModel(input_dim=4, hidden_dim=64, output_dim=output_dim, num_layers=2).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

if PREDICT_ROTATION:
    states = [{"TX": current_position[0],
               "TY": current_position[1],
               "TZ": current_position[2],
               "Rx": current_rotation[0],
               "Ry": current_rotation[1],
               "Rz": current_rotation[2]}]
else:
    states = [{"TX": current_position[0],
               "TY": current_position[1],
               "TZ": current_position[2]}]

for i in range(total_frames - SEQ_LENGTH):
    input_seq = (
        torch.tensor(expanded_sequence[i : i + SEQ_LENGTH], dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )
    with torch.no_grad():
        delta = model(input_seq).cpu().numpy().flatten()
    if PREDICT_ROTATION:
        current_position += delta[:3]
        current_rotation += delta[3:6]
        state = {
            "TX": current_position[0],
            "TY": current_position[1],
            "TZ": current_position[2],
            "Rx": current_rotation[0],
            "Ry": current_rotation[1],
            "Rz": current_rotation[2],
        }
    else:
        current_position += delta
        state = {
            "TX": current_position[0],
            "TY": current_position[1],
            "TZ": current_position[2],
        }
    states.append(state)

if PREDICT_ROTATION:
    columns = ["TX", "TY", "TZ", "Rx", "Ry", "Rz"]
else:
    columns = ["TX", "TY", "TZ"]
output_df = pd.DataFrame(states, columns=columns)
output_df["Frame"] = range(1, len(states) + 1)
output_df = output_df[["Frame"] + columns]
output_df.to_csv("predictions.csv", index=False)

for label in ["TX", "TY", "TZ"]:
    output_df["V_" + label] = output_df[label].diff().fillna(0)
    output_df["A_" + label] = output_df["V_" + label].diff().fillna(0)

fig, axs = plt.subplots(3, 1, figsize=(15, 18))
axs[0].set_title("Translation Position Over Time")
axs[0].set_xlabel("Frame")
axs[0].set_ylabel("Position")
for label in ["TX", "TY", "TZ"]:
    axs[0].plot(output_df["Frame"], output_df[label], label=label)
axs[0].legend()
axs[0].grid(True)

axs[1].set_title("Translation Velocity Over Time")
axs[1].set_xlabel("Frame")
axs[1].set_ylabel("Velocity")
for label in ["V_TX", "V_TY", "V_TZ"]:
    axs[1].plot(output_df["Frame"], output_df[label], label=label)
axs[1].legend()
axs[1].grid(True)

axs[2].set_title("Translation Acceleration Over Time")
axs[2].set_xlabel("Frame")
axs[2].set_ylabel("Acceleration")
for label in ["A_TX", "A_TY", "A_TZ"]:
    axs[2].plot(output_df["Frame"], output_df[label], label=label)
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("predictions_translation.png")
print("Translation predictions saved to predictions.csv and predictions_translation.png")

if PREDICT_ROTATION:
    for label in ["Rx", "Ry", "Rz"]:
        output_df["V_" + label] = output_df[label].diff().fillna(0)
        output_df["A_" + label] = output_df["V_" + label].diff().fillna(0)

    fig2, axs2 = plt.subplots(3, 1, figsize=(15, 18))
    axs2[0].set_title("Rotation Position Over Time")
    axs2[0].set_xlabel("Frame")
    axs2[0].set_ylabel("Rotation")
    for label in ["Rx", "Ry", "Rz"]:
        axs2[0].plot(output_df["Frame"], output_df[label], label=label)
    axs2[0].legend()
    axs2[0].grid(True)

    axs2[1].set_title("Rotation Velocity Over Time")
    axs2[1].set_xlabel("Frame")
    axs2[1].set_ylabel("Velocity")
    for label in ["V_Rx", "V_Ry", "V_Rz"]:
        axs2[1].plot(output_df["Frame"], output_df[label], label=label)
    axs2[1].legend()
    axs2[1].grid(True)

    axs2[2].set_title("Rotation Acceleration Over Time")
    axs2[2].set_xlabel("Frame")
    axs2[2].set_ylabel("Acceleration")
    for label in ["A_Rx", "A_Ry", "A_Rz"]:
        axs2[2].plot(output_df["Frame"], output_df[label], label=label)
    axs2[2].legend()
    axs2[2].grid(True)

    plt.tight_layout()
    plt.savefig("predictions_rotation.png")
    print("Rotation predictions plotted to predictions_rotation.png")
