import os, glob
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "processed_data"
EPOCHS, BATCH_SIZE, LR = 1000, 32, 1e-3
MODEL_SAVE_PATH = f"drone_movement_model_{int(EPOCHS/1000)}k.pt"
SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS = 30, 64, 1
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

        # scaling
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()

        self.input_scaler.fit(self.data[:, :9])
        joblib.dump(self.input_scaler, "input_scaler.pkl")

        self.output_scaler.fit(self.data[:, :6])
        joblib.dump(self.output_scaler, "output_scaler.pkl")

        self.data[:, :9] = self.input_scaler.transform(self.data[:, :9])
        self.data[:, 0:6] = self.output_scaler.transform(self.data[:, 0:6])

    def __len__(self):
        return len(self.data) - SEQ_LENGTH if len(self.data) >= SEQ_LENGTH else 0 #non-negative

    def __getitem__(self, idx):
        exogenous_inputs = self.data[idx:idx + SEQ_LENGTH, 6:9]
        past_outputs = self.data[idx:idx + SEQ_LENGTH, 0:6]
        inputs = np.concatenate([exogenous_inputs, past_outputs], axis=1)
        target = self.data[idx + SEQ_LENGTH, 0:6]                    
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class DroneNARXModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, output_dim=6, num_layers=1):
        super(DroneNARXModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])    

# training
dataset = DroneDataset(DATA_PATH, SEQ_LENGTH)
if len(dataset) == 0:
    raise ValueError("Dataset is empty. check csvs.")

# kinda not needed for now until we have enough data
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size,  len(dataset) - train_size])

loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

model = DroneNARXModel(input_dim=9, hidden_dim=HIDDEN_SIZE, output_dim=6, num_layers=NUM_LAYERS).to(DEVICE)
criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=LR)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True)

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

    avg_train_loss = total_loss / len(loader)

    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
    avg_validation_loss = validation_loss / len(validation_loader)

    scheduler.step(avg_validation_loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{EPOCHS} - loss: {avg_train_loss:.4f}, validation loss: {avg_validation_loss:.4f}")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f" model saved to {MODEL_SAVE_PATH}")
