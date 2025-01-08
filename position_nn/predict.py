import sys, os
import torch, torch.nn as nn
import numpy as np
import pandas as pd

MODEL_PATH = "drone_movement_model.pt"
SEQ_LENGTH = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <input_file.csv>")
    sys.exit(1)

input_file = sys.argv[1]
model = DroneMovementModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

df = pd.read_csv(input_file)
input_seq = torch.tensor(df[['potX', 'potY', 'potZ']].values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
prediction = model(input_seq).cpu().detach().numpy().flatten()

labels = ["RX", "RY", "RZ", "TX", "TY", "TZ"]
print("Predicted Drone Movement [[RX, RY, RZ, TX, TY, TZ]]:")
for label, value in zip(labels, prediction):
    print(f"{label}: {value:.2f}")
