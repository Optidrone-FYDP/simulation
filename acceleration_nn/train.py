import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np

with open("out/accelerations.json", "r") as f:
    result_mapping = json.load(f)

X = []
y = []

for key, value in result_mapping.items():
    pot_x = int(key[0:3])
    pot_y = int(key[3:6])
    pot_z = int(key[6:9])
    avg_acc = value
    
    X.append([pot_x, pot_y, pot_z])
    y.append(avg_acc)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# normalize inputs
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / (X_std + 1e-8) # small number to avoid division by 0

# Split into train/test sets
train_ratio = 0.8
num_samples = len(X_norm)
split_index = int(num_samples * train_ratio)

X_train = X_norm[:split_index]
y_train = y[:split_index]

X_val = X_norm[split_index:]
y_val = y[split_index:]

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)

X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleMLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    # validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_loss = criterion(val_pred, y_val_t)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

normalization_params = {
    "X_mean": X_mean.tolist(),
    "X_std": X_std.tolist()
}

with open("normalization_params.json", "w") as f:
    json.dump(normalization_params, f)

torch.save(model.state_dict(), "model.pth")
