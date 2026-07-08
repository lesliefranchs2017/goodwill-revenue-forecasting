# Goodwill PyTorch 3000-row no-income model
# Parallel structure for TensorFlow comparison
# Updated July 8, 2026

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "goodwill_mock_data_3000_revised_noincome.csv"
OUTPUT_CSV = "pytorch_revenue_predictions_3000_noincome.csv"
PLOT_FILE = "pytorch_actual_vs_predicted_3000_noincome.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Match TensorFlow feature order exactly
feature_cols = ["Expenses", "Unemployment", "CPI", "BarrelOil", "NetAssets"]
target_col = "Revenue"

# Reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


print("\nGoodwill PyTorch 3000-row no-income model")
print("Parallel structure for TensorFlow comparison")
print("-" * 60)


# -----------------------------
# Load and clean data
# -----------------------------
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

required_cols = feature_cols + [target_col]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols).copy()

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32).reshape(-1, 1)

print(f"Rows after cleaning: {len(df):,}")
print(f"Features: {feature_cols}")
print(f"Target: {target_col}")


# -----------------------------
# 80/20 train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")


# -----------------------------
# Scale X and y
# Important: fit scalers on training data only
# -----------------------------
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)


# -----------------------------
# PyTorch model
# Match TensorFlow architecture:
# Dense(16, relu) -> Dense(8, relu) -> Dense(1)
# -----------------------------
class GoodwillRevenueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


model = GoodwillRevenueModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -----------------------------
# Train model
# -----------------------------
start_time = time.time()

print("\nTraining model...")
for epoch in range(EPOCHS):
    model.train()

    permutation = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train_tensor.size(0), BATCH_SIZE):
        batch_indices = permutation[i:i + BATCH_SIZE]
        batch_X = X_train_tensor[batch_indices]
        batch_y = y_train_tensor[batch_indices]

        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_X.size(0)

    epoch_loss = epoch_loss / X_train_tensor.size(0)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            test_pred_scaled = model(X_test_tensor)
            test_loss = criterion(test_pred_scaled, y_test_tensor).item()
        print(
            f"Epoch {epoch + 1:>3}/{EPOCHS} | "
            f"Train Loss: {epoch_loss:.6f} | "
            f"Test Loss: {test_loss:.6f}"
        )

training_time = time.time() - start_time


# -----------------------------
# Evaluate on test set only
# -----------------------------
model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(X_test_tensor).numpy()
    y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled)

    y_pred_train_scaled = model(X_train_tensor).numpy()
    y_pred_train = y_scaler.inverse_transform(y_pred_train_scaled)


test_mae = mean_absolute_error(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_pred_test)

train_mae = mean_absolute_error(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_pred_train)


# -----------------------------
# Future prediction
# Feature order must match feature_cols:
# Expenses, Unemployment, CPI, BarrelOil, NetAssets
# -----------------------------
x_new = np.array([[70000000, 5.3, 250.142, 93, 43309235.5333]], dtype=np.float32)
x_new_scaled = X_scaler.transform(x_new)
x_new_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)

with torch.no_grad():
    y_new_pred_scaled = model(x_new_tensor).numpy()
    y_new_pred = y_scaler.inverse_transform(y_new_pred_scaled)
    y_new_pred_value = float(y_new_pred[0][0])


# -----------------------------
# Save predictions
# -----------------------------
test_output = pd.DataFrame(X_test, columns=feature_cols)
test_output["Actual_Revenue"] = y_test.flatten()
test_output["Predicted_Revenue"] = y_pred_test.flatten()
test_output["Absolute_Error"] = abs(test_output["Actual_Revenue"] - test_output["Predicted_Revenue"])
test_output["Dataset"] = "Test"

train_output = pd.DataFrame(X_train, columns=feature_cols)
train_output["Actual_Revenue"] = y_train.flatten()
train_output["Predicted_Revenue"] = y_pred_train.flatten()
train_output["Absolute_Error"] = abs(train_output["Actual_Revenue"] - train_output["Predicted_Revenue"])
train_output["Dataset"] = "Train"

future_output = pd.DataFrame(x_new, columns=feature_cols)
future_output["Actual_Revenue"] = np.nan
future_output["Predicted_Revenue"] = y_new_pred_value
future_output["Absolute_Error"] = np.nan
future_output["Dataset"] = "Future"

final_output = pd.concat([train_output, test_output, future_output], ignore_index=True)
final_output.to_csv(OUTPUT_CSV, index=False)


# -----------------------------
# Plot actual vs predicted for test rows
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.flatten(), label="Actual Revenue")
plt.plot(range(len(y_pred_test)), y_pred_test.flatten(), label="Predicted Revenue")
plt.xlabel("Test Row Index")
plt.ylabel("Revenue")
plt.title("PyTorch Actual vs Predicted Revenue - Test Data")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=150)
plt.show()


# -----------------------------
# PowerPoint-ready summary
# -----------------------------
print("\n")
print("=" * 60)
print("        Goodwill Revenue Prediction Summary")
print("=" * 60)

print(f"Dataset                : {CSV_PATH}")
print(f"Rows                   : {len(df):,}")
print(f"Features               : {len(feature_cols)}")
print(f"Target                 : {target_col}")

print("\nData Split")
print("-" * 60)
print(f"Training Rows          : {len(X_train):,}")
print(f"Testing Rows           : {len(X_test):,}")
print(f"Split                  : 80% train / 20% test")
print(f"Random State           : {RANDOM_STATE}")

print("\nModel")
print("-" * 60)
print("Framework              : PyTorch")
print("Model                  : Neural Network")
print("Architecture           : 5 inputs -> 16 ReLU -> 8 ReLU -> 1 output")
print("Optimizer              : Adam")
print("Loss Function          : Mean Squared Error (MSE)")
print(f"Epochs                 : {EPOCHS}")
print(f"Batch Size             : {BATCH_SIZE}")
print(f"Learning Rate          : {LEARNING_RATE}")

print("\nInput Features")
print("-" * 60)
for feature in feature_cols:
    print(f"   - {feature}")

print("\nPerformance - Test Data")
print("-" * 60)
print(f"Test MAE               : {test_mae:,.2f}")
print(f"Test MSE               : {test_mse:,.2f}")
print(f"Test RMSE              : {test_rmse:,.2f}")
print(f"Test R²                : {test_r2:.4f}")

print("\nPerformance - Training Data")
print("-" * 60)
print(f"Train MAE              : {train_mae:,.2f}")
print(f"Train MSE              : {train_mse:,.2f}")
print(f"Train RMSE             : {train_rmse:,.2f}")
print(f"Train R²               : {train_r2:.4f}")

print("\nPrediction")
print("-" * 60)
print(f"Future Revenue         : ${y_new_pred_value:,.2f}")

print("\nFiles Created")
print("-" * 60)
print(f"Prediction CSV         : {OUTPUT_CSV}")
print(f"Plot File              : {PLOT_FILE}")

print("\nRuntime")
print("-" * 60)
print(f"Training Time          : {training_time:.2f} seconds")

print("=" * 60)
print("End of Model")
print("=" * 60)
