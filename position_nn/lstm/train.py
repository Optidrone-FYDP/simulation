import os, glob
import pandas as pd
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PREDICT_ROTATION = True
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import random

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

PREDICT_ROTATION = False

DATA_PATH = "../final_train_set"
MODEL_SAVE_PATH = "models/drone_movement_model_lstm_0.9.pt"
EPOCHS, BATCH_SIZE, LR = 100, 32, 1e-3
SEQ_LENGTH, HIDDEN_SIZE, NUM_LAYERS = 20, 64, 2
TRAIN_SPLIT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROT_NORM_FACTOR = 2 * np.pi

def normalize_pot(x):
    """Normalize potentiometer value: from [0, 128] to [-1, 1]."""
    return (x - 64.0) / 64.0

class DroneDataset(Dataset):
    def __init__(self, folder):
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        dfs = [
            pd.read_csv(f, encoding="utf-8-sig").values.astype(np.float32)
            for f in files
        ]
        self.data = np.concatenate(dfs, axis=0) if dfs else np.empty((0, 11))
        if self.data.size:
            self.data[:, 10] = normalize_pot(self.data[:, 10])  # potX
            self.data[:, 9] = normalize_pot(self.data[:, 9])    # potY
            self.data[:, 7] = normalize_pot(self.data[:, 7])    # potZ
        print("Dataset shape:", self.data.shape)

    def __len__(self):
        return len(self.data) - SEQ_LENGTH if len(self.data) >= SEQ_LENGTH else 0

    def __getitem__(self, idx):
        sequence = self.data[idx : idx + SEQ_LENGTH]
        inputs = sequence[:, [10, 9, 7]]
        
        if PREDICT_ROTATION:
            trans_delta = self.data[idx + SEQ_LENGTH, [4, 5, 6]] - self.data[idx + SEQ_LENGTH - 1, [4, 5, 6]]
            target = np.concatenate([trans_delta])
        else:
            target = self.data[idx + SEQ_LENGTH, [4, 5, 6]] - self.data[idx + SEQ_LENGTH - 1, [4, 5, 6]]
        
        return torch.tensor(inputs), torch.tensor(target)

class DroneMovementModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=2):
        super(DroneMovementModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# adjust model output dimension
output_dim = 6 if PREDICT_ROTATION else 3

dataset = DroneDataset(DATA_PATH)
if len(dataset) == 0:
    raise ValueError("Dataset empty, check data path")

train_size = int(TRAIN_SPLIT * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

model = DroneMovementModel(input_dim=3, hidden_dim=HIDDEN_SIZE, output_dim=output_dim, num_layers=NUM_LAYERS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

patience = 80
best_validation_loss = float("inf")
epochs_without_improvement = 0
best_model_state = None

train_losses = []
validation_losses = []
epochs_record = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    total_mae = 0.0
    for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = torch.sqrt(criterion(outputs, targets))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        mae = torch.mean(torch.abs(outputs - targets))
        total_mae += mae.item()
    avg_train_loss = total_loss / len(train_loader)
    avg_train_mae = total_mae / len(train_loader)

    model.eval()
    validation_loss = 0
    validation_mae = 0.0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = torch.sqrt(criterion(outputs, targets))
            validation_loss += loss.item()

            mae = torch.mean(torch.abs(outputs - targets))
            validation_mae += mae.item()

    avg_validation_mae = validation_mae / len(validation_loader)
    avg_validation_loss = validation_loss / len(validation_loader)

    scheduler.step()
    train_losses.append(avg_train_loss)
    validation_losses.append(avg_validation_loss)
    epochs_record.append(epoch)

    if avg_validation_loss < best_validation_loss:
        best_validation_loss = avg_validation_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict()
        torch.save({
            "model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "hyperparameters": {
                "SEQ_LENGTH": SEQ_LENGTH,
                "HIDDEN_SIZE": HIDDEN_SIZE,
                "NUM_LAYERS": NUM_LAYERS,
                "LR": LR,
                "BATCH_SIZE": BATCH_SIZE,
            },
            "rotation_norm_factor": ROT_NORM_FACTOR,
        }, MODEL_SAVE_PATH)
    else:
        epochs_without_improvement += 1

    print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {best_validation_loss:.4f}\n Train MAE: {avg_train_mae:.4f} mm, Validation MAE: {avg_validation_mae:.4f} mm")

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch}. Best Validation Loss: {best_validation_loss:.4f}")
        break

print(f"Best model saved to {MODEL_SAVE_PATH}")

plt.figure(figsize=(10, 5))
plt.plot(epochs_record, train_losses, label="Train Loss")
plt.plot(epochs_record, validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.savefig("loss.png")
