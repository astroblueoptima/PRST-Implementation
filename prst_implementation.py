
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# 1. Data Loading and Preprocessing
btc_data = pd.read_csv('BTC-USD 5.csv')
features = btc_data[["Open", "High", "Low", "Close", "Volume"]].values[:-1]
target = btc_data["Close"].shift(-1).dropna().values

def normalize_data(data):
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    return (data - min_vals) / (max_vals - min_vals)

X_normalized = normalize_data(features)
y_min, y_max = target.min(), target.max()
y_normalized = (target - y_min) / (y_max - y_min)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

# 2. Model Definition
class PRSTModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=16, dropout_prob=0.2):
        super(PRSTModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.output(x)

# 3. Training with PRST
def prst_train(model, X, y, optimizer, criterion, epochs=50, batch_size=32, scarcity_factor=0.5):
    train_losses = []
    
    for epoch in range(epochs):
        # Data Scarcity: Randomly sample a portion of the training data
        indices = torch.randperm(len(X))[:int(scarcity_factor * len(X))]
        inputs, targets = X[indices], y[indices]

        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Adjust scarcity factor based on performance (Dynamic Adaptation)
        if epoch > 0 and train_losses[-1] > train_losses[-2]:
            scarcity_factor = min(1.0, scarcity_factor + 0.1)
        else:
            scarcity_factor = max(0.1, scarcity_factor - 0.1)

    return model

# Initialize and train the model using PRST
model = PRSTModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

trained_model = prst_train(model, X_tensor, y_tensor, optimizer, criterion)
