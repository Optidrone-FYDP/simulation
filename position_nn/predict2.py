import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

class DroneNARXModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneNARXModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

MODEL_PATH = "drone_movement_model_1k.pt"
INPUT_SCALER_PATH = "input_scaler.pkl"
OUTPUT_SCALER_PATH = "output_scaler.pkl"
SEQ_LENGTH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def expand_pot_inputs(df):
    sequence = []
    for _, row in df.iterrows():
        pot = [row['potX'], row['potY'], row['potZ']]
        duration = int(row['duration'])
        sequence.extend([pot] * duration)
    return np.array(sequence, dtype=np.float32)

if len(sys.argv) != 8:
    print(f"Usage: python {sys.argv[0]} <simulation_inputs.csv> <RX> <RY> <RZ> <TX> <TY> <TZ>")
    sys.exit(1)

input_file = sys.argv[1]
start_RX, start_RY, start_RZ, start_TX, start_TY, start_TZ = map(float, sys.argv[2:8])

if not os.path.exists(input_file):
    print(f"input file {input_file} does not exist")
    sys.exit(1)

df_input = pd.read_csv(input_file)
expanded_sequence = expand_pot_inputs(df_input)
total_frames = len(expanded_sequence)
print(f"expanded sequence: {expanded_sequence.shape}")

# Load scalers
if not os.path.exists(INPUT_SCALER_PATH) or not os.path.exists(OUTPUT_SCALER_PATH):
    print("error: scalar files not found")
    sys.exit(1)

input_scaler = joblib.load(INPUT_SCALER_PATH)
output_scaler = joblib.load(OUTPUT_SCALER_PATH)
print("scalers loaded")

# Load  model
if not os.path.exists(MODEL_PATH):
    print(f"model file {MODEL_PATH} not found")
    sys.exit(1)

model = DroneNARXModel().to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("model loaded")

current_position = np.array([start_RX, start_RY, start_RZ, start_TX, start_TY, start_TZ], dtype=np.float32)
positions = [current_position.copy()]

initial_sequence = []
for i in range(SEQ_LENGTH):
    exog_input = expanded_sequence[i]  # [potX, potY, potZ]
    combined = np.concatenate([exog_input, current_position])  # [potX, potY, potZ, RX, RY, RZ, TX, TY, TZ]
    initial_sequence.append(combined)
initial_sequence_ = np.array(initial_sequence, dtype=np.float32)
print(f"Initial sequence shape (): {initial_sequence_.shape}")

# scale initial_sequence
initial_sequence_scaled = input_scaler.transform(initial_sequence)

# convert to tensor
input_seq = torch.tensor(initial_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# prediction loop
for i in range(SEQ_LENGTH, total_frames):
    with torch.no_grad():
        pred_scaled_ = model(input_seq)
    pred = output_scaler.inverse_transform(pred_scaled_.cpu().numpy())
    pred = pred.flatten()
    print(f"Frame {i+1}:  Prediction: {pred}")

    current_position = pred
    positions.append(current_position.copy())

    exogenous_input = expanded_sequence[i]
    combined = np.concatenate([exogenous_input, current_position])
    combined_scaled = input_scaler.transform(combined.reshape(1, -1))

    combined_scaled = combined_scaled.flatten()
    print(f"Combined scaled shape (): {combined_scaled.shape}")

    input_seq_np = input_seq.cpu().numpy()
    input_seq_np = np.concatenate([input_seq_np[:, 1:, :], combined_scaled.reshape(1, 1, -1)], axis=1)
    input_seq = torch.tensor(input_seq_np, dtype=torch.float32).to(DEVICE)

labels = ["RX", "RY", "RZ", "TX", "TY", "TZ"]
output_df = pd.DataFrame(positions, columns=labels)
output_df['Frame'] = range(1, len(positions)+1)
output_df = output_df[['Frame'] + labels]

output_df.to_csv("predictions.csv", index=False)

output_df[['V_RX', 'V_RY', 'V_RZ', 'V_TX', 'V_TY', 'V_TZ']] = output_df[labels].diff().fillna(0)
output_df[['A_RX', 'A_RY', 'A_RZ', 'A_TX', 'A_TY', 'A_TZ']] = output_df[['V_RX', 'V_RY', 'V_RZ', 'V_TX', 'V_TY', 'V_TZ']].diff().fillna(0)

fig, axs = plt.subplots(3, 1, figsize=(15, 18))

for label in labels:
    axs[0].plot(output_df['Frame'], output_df[label], label=label)
axs[0].set_title('Position Over Time')
axs[0].set_xlabel('Frame')
axs[0].set_ylabel('Position')
axs[0].legend()
axs[0].grid(True)

for label in ['V_RX', 'V_RY', 'V_RZ', 'V_TX', 'V_TY', 'V_TZ']:
    axs[1].plot(output_df['Frame'], output_df[label], label=label)
axs[1].set_title('Velocity Over Time')
axs[1].set_xlabel('Frame')
axs[1].set_ylabel('Velocity')
axs[1].legend()
axs[1].grid(True)

for label in ['A_RX', 'A_RY', 'A_RZ', 'A_TX', 'A_TY', 'A_TZ']:
    axs[2].plot(output_df['Frame'], output_df[label], label=label)
axs[2].set_title('Acceleration Over Time')
axs[2].set_xlabel('Frame')
axs[2].set_ylabel('Acceleration')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("predictions.png")
