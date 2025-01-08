import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DroneDataset(Dataset):
    def __init__(self, data, joystick_inputs, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.joystick_inputs = torch.tensor(joystick_inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.joystick_inputs[idx], self.targets[idx]

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_dim = 9
output_dim = 6
learning_rate = 0.001
batch_size = 32
epochs = 50

# Initialize model
model = SimpleNN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Prepare dataset
train_dataset = DroneDataset(train_data, train_joystick_inputs, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for inputs, joystick_inputs, targets in train_loader:
        optimizer.zero_grad()
        predictions = model(torch.cat((inputs, joystick_inputs), dim=1))
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
