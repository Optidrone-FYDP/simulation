import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

MODEL_PATH = "models/drone_movement_model_lstm_0.9.pt"
SEQ_LENGTH = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

def normalize_pot(x):
    return (x - 64.0) / 64.0

class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=2):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def expand_pot_inputs(df):
    sequence = []
    for _, row in df.iterrows():
        pot = [normalize_pot(row["potX"]),
               normalize_pot(row["potY"]),
               normalize_pot(row["potZ"])]
        duration = int(row["duration"])
        sequence.extend([pot] * duration)
    return np.array(sequence, dtype=np.float32)

if len(sys.argv) != 5:
    print(f"Usage: python {sys.argv[0]} <simulation_inputs.csv> <start_TX> <start_TY> <start_TZ>")
    sys.exit(1)

input_file = sys.argv[1]
start_TX, start_TY, start_TZ = map(float, sys.argv[2:5])
current_position = np.array([start_TX, start_TY, start_TZ], dtype=np.float32)

df_input = pd.read_csv(input_file)
expanded_sequence = expand_pot_inputs(df_input)
total_frames = len(expanded_sequence)

model = DroneMovementModel(input_dim=3, hidden_dim=64, output_dim=3, num_layers=2).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)
model.eval()

states = [{"TX": current_position[0],
           "TY": current_position[1],
           "TZ": current_position[2]}]

for i in range(total_frames - SEQ_LENGTH):
    input_seq = torch.tensor(expanded_sequence[i : i + SEQ_LENGTH], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        delta = model(input_seq).cpu().numpy().flatten()
    current_position += delta
    state = {"TX": current_position[0],
             "TY": current_position[1],
             "TZ": current_position[2]}
    states.append(state)

output_df = pd.DataFrame(states)
output_df["Frame"] = range(1, len(states) + 1)
output_df = output_df[["Frame", "TX", "TY", "TZ"]]
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
