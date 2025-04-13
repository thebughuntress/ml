import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(
    "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/BostonHousing_train.csv"
)
print(df.sample(5))

# Data separation
y = torch.tensor(df["medv"].values, dtype=torch.float32).view(-1, 1)
X = torch.tensor(df.drop("medv", axis=1).values, dtype=torch.float32)
print(y[:5])
print(X[:5])

# Data splitting
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    """Simple train-test split using PyTorch and Pandas."""
    if random_state is not None:
        torch.manual_seed(random_state)
    indices = torch.randperm(X.shape[0])
    test_len = int(X.shape[0] * test_size)
    test_indices = indices[:test_len]
    train_indices = indices[test_len:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=100)


# Linear Regression Model (PyTorch)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)  # 1 output for medv

    def forward(self, x):
        return self.linear(x)


input_size = X_train.shape[1]
lr_model = LinearRegressionModel(input_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(lr_model.parameters(), lr=0.0001)  # Adjust learning rate

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    y_lr_train_pred = lr_model(X_train)
    loss = criterion(y_train, y_lr_train_pred)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predictions
with torch.no_grad():
    y_lr_train_pred = lr_model(X_train)
    y_lr_test_pred = lr_model(X_test)


# Custom MSE and R^2 functions (PyTorch)
def custom_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def custom_r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# Evaluate Linear Regression
lr_train_mse = custom_mse(y_train, y_lr_train_pred).item()
lr_train_r2 = custom_r2_score(y_train, y_lr_train_pred).item()
lr_test_mse = custom_mse(y_test, y_lr_test_pred).item()
lr_test_r2 = custom_r2_score(y_test, y_lr_test_pred).item()

print("Linear Regression Performance:")
print(f"Train MSE: {lr_train_mse:.4f}")
print(f"Train R² : {lr_train_r2:.4f}")
print(f"Test MSE : {lr_test_mse:.4f}")
print(f"Test R²  : {lr_test_r2:.4f}")

lr_results = pd.DataFrame(
    ["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]
).transpose()
lr_results.columns = ["Model", "Training MSE", "Training R²", "Test MSE", "Test R²"]
print(lr_results)


# Random Forest (PyTorch) - Simplified (Not a full RF implementation)
class SimpleRandomForest(nn.Module):
    def __init__(self, input_size, num_trees, max_depth):
        super(SimpleRandomForest, self).__init__()
        self.trees = nn.ModuleList([SimpleDecisionTree(input_size, max_depth) for _ in range(num_trees)])

    def forward(self, x):
        predictions = torch.stack([tree(x) for tree in self.trees])
        return torch.mean(predictions, dim=0)


class SimpleDecisionTree(nn.Module):
    def __init__(self, input_size, max_depth):
        super(SimpleDecisionTree, self).__init__()
        self.max_depth = max_depth
        self.splits = nn.Parameter(torch.randn(max_depth, input_size))  # Simplified: Random splits
        self.thresholds = nn.Parameter(torch.randn(max_depth))
        self.values = nn.Parameter(torch.randn(2**max_depth)) # Simplified: Leaf values

    def forward(self, x):
        batch_size = x.size(0)
        indices = torch.arange(batch_size)
        out = torch.zeros(batch_size, 1, device=x.device)
        self._recursive_forward(x, indices, 0, 0, out)
        return out

    def _recursive_forward(self, x, active_indices, depth, node_idx, out):
        if depth == self.max_depth:
            leaf_idx = node_idx - 2**self.max_depth
            out[active_indices] = self.values[leaf_idx]
            return

        split_idx = depth % self.max_depth
        split_feature = self.splits[split_idx]
        threshold = self.thresholds[split_idx]

        left_mask = x[active_indices] @ split_feature <= threshold
        right_mask = ~left_mask

        left_indices = active_indices[left_mask]
        right_indices = active_indices[right_mask]

        if left_indices.numel() > 0:
            self._recursive_forward(x, left_indices, depth + 1, 2 * node_idx + 1, out)
        if right_indices.numel() > 0:
            self._recursive_forward(x, right_indices, depth + 1, 2 * node_idx + 2, out)


# Simplified RF model instantiation and training
num_trees = 10
max_depth = 3
rf_model = SimpleRandomForest(X_train.shape[1], num_trees, max_depth)
rf_criterion = nn.MSELoss()
rf_optimizer = optim.SGD(rf_model.parameters(), lr=0.001)

# Training loop
rf_epochs = 50
for epoch in range(rf_epochs):
    y_rf_train_pred = rf_model(X_train)
    rf_loss = rf_criterion(y_train, y_rf_train_pred)
    rf_optimizer.zero_grad()
    rf_loss.backward()
    rf_optimizer.step()

# Predictions
with torch.no_grad():
    y_rf_train_pred = rf_model(X_train)
    y_rf_test_pred = rf_model(X_test)

# Evaluate Random Forest (PyTorch)
rf_train_mse = custom_mse(y_train, y_rf_train_pred).item()
rf_train_r2 = custom_r2_score(y_train, y_rf_train_pred).item()
rf_test_mse = custom_mse(y_test, y_rf_test_pred).item()
rf_test_r2 = custom_r2_score(y_test, y_rf_test_pred).item()

print("\nRandom Forest Performance (PyTorch):")
print(f"Train MSE: {rf_train_mse:.4f}")
print(f"Train R² : {rf_train_r2:.4f}")
print(f"Test MSE : {rf_test_mse:.4f}")
print(f"Test R²  : {rf_test_r2:.4f}")

rf_results = pd.DataFrame(
    ["Random Forest", rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]
).transpose()
rf_results.columns = ["Model", "Training MSE", "Training R²", "Test MSE", "Test R²"]
print(rf_results)


# Model Comparison
df_models = pd.concat([rf_results, lr_results], axis=0)
df_models = df_models.reset_index(drop=True)
print(df_models)


# Data visualization

# Linear Regression (PyTorch)
plt.figure(figsize=(5, 5))
plt.scatter(x=y_train.numpy(), y=y_lr_train_pred.detach().numpy(), c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train.numpy().ravel(), y_lr_train_pred.detach().numpy().ravel(), 1)
p = np.poly1d(z)
plt.plot(y_train.numpy(), p(y_train.numpy()), "#F8766D")
plt.title("Actual vs Predicted with Linear Regression (PyTorch)")
plt.ylabel("Predicted Value")
plt.xlabel("Actual Value")
plt.show()


# Random Forest (PyTorch)
plt.figure(figsize=(5, 5))
plt.scatter(x=y_train.numpy(), y=y_rf_train_pred.detach().numpy(), c="#7CAE00", alpha=0.3)
z = np.polyfit(y_train.numpy().ravel(), y_rf_train_pred.detach().numpy().ravel(), 1)
p = np.poly1d(z)
plt.plot(y_train.numpy(), p(y_train.numpy()), "#F8766D")
plt.title("Actual vs Predicted with Random Forest (PyTorch)")
plt.ylabel("Predicted Value")
plt.xlabel("Actual Value")
plt.show()


# Example of Predicting a Single Value
crim = 0.35809
zn = 0.0
indus = 6.2
chas = 1
nox = 0.507
rm = 6.951
age = 88.5
dis = 2.8617
rad = 8
tax = 307
ptratio = 17.4
b = 391.7
lstat = 9.71
medv = 26.7

X_new = torch.tensor(np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]]), dtype=torch.float32)

with torch.no_grad():
    y_pred_lr_single = lr_model(X_new).item()
    y_pred_rf_single = rf_model(X_new).item()  # Use the PyTorch RF model

print("Actual value:", medv)
print("Predicted Value (Linear Regression):", y_pred_lr_single)
print("Predicted Value (Random Forest):", y_pred_rf_single)