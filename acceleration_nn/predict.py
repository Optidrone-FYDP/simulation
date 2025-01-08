import sys
import torch
import torch.nn as nn
import numpy as np
import json

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

if len(sys.argv) != 4:
    print("Usage: python predict.py <pot_x> <pot_y> <pot_z>")
    sys.exit(1)

# parse CLI arguments
try:
    pot_x = int(sys.argv[1])
    pot_y = int(sys.argv[2])
    pot_z = int(sys.argv[3])
except ValueError:
    print("Error: All pot values must be numbers.")
    sys.exit(1)

# load normalization params
with open("normalization_params.json", "r") as f:
    params = json.load(f)
X_mean = np.array(params["X_mean"], dtype=np.float32)
X_std = np.array(params["X_std"], dtype=np.float32)

# load model
model = SimpleMLP()
model.load_state_dict(torch.load("model.pth", map_location='cpu', weights_only=True))
model.eval()

# normalize input in same way as training
new_input = np.array([[pot_x,pot_y, pot_z]], dtype=np.float32)
new_input_norm = (new_input - X_mean) / (X_std + 1e-8)
new_input_t = torch.tensor(new_input_norm)

# predict
with torch.no_grad():
    prediction = model(new_input_t).numpy()[0]

print("Predicted acceleration (steady-state):")
print(f"X: {prediction[0]:.2f} mm/s^2")
print(f"Y: {prediction[1]:.2f} mm/s^2")
print(f"Z: {prediction[2]:.2f} mm/s^2")
