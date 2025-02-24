import os, glob
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "processed_data"
MODEL_SAVE_PATH = "drone_movement_model_5k.pt"
EPOCHS, BATCH_SIZE, LR = 5000, 32, 1e-3
SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS = 10, 64, 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DroneDataset(Dataset):
    def __init__(self, folder, _):
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        dfs = [ pd.read_csv(f).values.astype(np.float32) for f in files ]
        if dfs:
            self.data = np.concatenate(dfs, axis=0)
        else:
            self.data = np.empty((0, 9))
        print(self.data)

    def __len__(self):
        return len(self.data) - SEQ_LENGTH if len(self.data) >= SEQ_LENGTH else 0 #non-negative

    def __getitem__(self, idx):
        return self.data[idx:idx+SEQ_LENGTH, 6:9], self.data[idx+SEQ_LENGTH, 0:6]

class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# training
dataset = DroneDataset(DATA_PATH, SEQ_LENGTH)
if len(dataset) == 0:
    raise ValueError("Dataset is empty. check csvs.")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
model = DroneMovementModel(input_dim=3, hidden_dim=HIDDEN_SIZE, output_dim=6, num_layers=NUM_LAYERS).to(DEVICE)
criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR)

# training loop
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for i, (inputs, targets) in enumerate(loader, 1):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Step {i}, Loss: {loss.item():.4f}")
    print(f"Epoch {epoch}/{EPOCHS} - Avg Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
