# Goodwill TensorFlow 3000-row no-income model
# Parallel structure for PyTorch comparison
# Updated for clean 80/20 review output

import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\nGoodwill TensorFlow 3000-row no-income model")
print("Parallel structure for PyTorch comparison")
print("TensorFlow Version:", tf.__version__)

# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
csv_path = "goodwill_mock_data_3000_revised_noincome.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

feature_cols = ["Expenses", "CPI", "Unemployment", "BarrelOil", "NetAssets"]
target_col = "Revenue"
required_cols = feature_cols + [target_col]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

# Force numeric and remove bad rows
df = df.copy()
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols)

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32).reshape(-1, 1)

if not np.isfinite(X).all():
    raise ValueError("X contains NaN or infinite values.")
if not np.isfinite(y).all():
    raise ValueError("y contains NaN or infinite values.")

print(f"Rows after cleaning: {len(df):,}")
print(f"Features: {feature_cols}")
print(f"Target: {target_col}")

# -------------------------------------------------------------------
# 2. Train/test split: same 80/20 structure as PyTorch
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

print(f"Training rows: {len(X_train):,}")
print(f"Test rows: {len(X_test):,}")

# -------------------------------------------------------------------
# 3. Standardize using TRAINING data only
#    This avoids letting test data influence the training scale.
# -------------------------------------------------------------------
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True)
X_std[X_std == 0] = 1.0

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Scale y for more stable neural-network training, then convert back later.
y_mean = y_train.mean(axis=0, keepdims=True)
y_std = y_train.std(axis=0, keepdims=True)
y_std[y_std == 0] = 1.0

y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# -------------------------------------------------------------------
# 4. Build TensorFlow model
# -------------------------------------------------------------------
epochs = 500
batch_size = 32
learning_rate = 0.01

model = Sequential([
    Input(shape=(len(feature_cols),)),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mse",
    metrics=["mae"]
)

print("\nTraining TensorFlow model...")
start_time = time.time()

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.20,
    verbose=0
)

training_time = time.time() - start_time

# -------------------------------------------------------------------
# 5. Evaluate on TEST rows only
# -------------------------------------------------------------------
y_pred_scaled = model.predict(X_test_scaled, verbose=0)
y_pred_actual = y_pred_scaled * y_std + y_mean
y_test_actual = y_test

test_mae = mean_absolute_error(y_test_actual, y_pred_actual)
test_mse = mean_squared_error(y_test_actual, y_pred_actual)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test_actual, y_pred_actual)

# TensorFlow scaled metrics from model.evaluate, useful but not the deck metric.
test_loss_scaled, test_mae_scaled = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)

# -------------------------------------------------------------------
# 6. Future prediction row
# -------------------------------------------------------------------
x_new = np.array([[70000000, 250.142, 5.3, 93, 43309235.5333]], dtype=np.float32)
x_new_scaled = (x_new - X_mean) / X_std
y_new_pred_scaled = model.predict(x_new_scaled, verbose=0)
y_new_pred = y_new_pred_scaled * y_std + y_mean
y_new_pred_value = float(y_new_pred[0][0])

# -------------------------------------------------------------------
# 7. Save output comparison file
# -------------------------------------------------------------------
output_df = pd.DataFrame({
    "Actual_Revenue": y_test_actual.flatten(),
    "Predicted_Revenue": y_pred_actual.flatten(),
    "Difference": (y_test_actual.flatten() - y_pred_actual.flatten()),
    "Absolute_Error": np.abs(y_test_actual.flatten() - y_pred_actual.flatten())
})

output_csv = "tensorflow_test_predictions_3000_noincome.csv"
output_df.to_csv(output_csv, index=False)

# -------------------------------------------------------------------
# 8. Clean summary for PowerPoint / review
# -------------------------------------------------------------------
print("\n")
print("=" * 60)
print("        Goodwill Revenue Prediction Summary")
print("=" * 60)

print(f"Dataset                : {csv_path}")
print(f"Rows                   : {len(df):,}")
print(f"Features               : {len(feature_cols)}")
print(f"Target                 : {target_col}")

print("\nData Split")
print("-" * 60)
print(f"Training Rows          : {len(X_train):,}")
print(f"Testing Rows           : {len(X_test):,}")
print("Split                  : 80% train / 20% test")
print("Random State           : 42")

print("\nModel")
print("-" * 60)
print("Framework              : TensorFlow")
print("Model                  : Dense Neural Network")
print("Hidden Layers          : 16 ReLU, 8 ReLU")
print("Output Layer           : 1 Revenue prediction")
print("Optimizer              : Adam")
print("Loss Function          : Mean Squared Error (MSE)")
print(f"Epochs                 : {epochs}")
print(f"Batch Size             : {batch_size}")
print(f"Learning Rate          : {learning_rate}")

print("\nInput Features")
print("-" * 60)
for feature in feature_cols:
    print(f"  - {feature}")

print("\nPerformance on 600 Test Rows")
print("-" * 60)
print(f"Test MAE               : ${test_mae:,.2f}")
print(f"Test MSE               : {test_mse:,.2f}")
print(f"Test RMSE              : ${test_rmse:,.2f}")
print(f"Test R²                : {test_r2:.4f}")
print(f"Scaled Test Loss       : {test_loss_scaled:.6f}")
print(f"Scaled Test MAE        : {test_mae_scaled:.6f}")

print("\nPrediction")
print("-" * 60)
print(f"Future Revenue         : ${y_new_pred_value:,.2f}")

print("\nFiles Created")
print("-" * 60)
print(f"Test Predictions CSV   : {output_csv}")

print("\nRuntime")
print("-" * 60)
print(f"Training Time          : {training_time:.2f} seconds")

print("=" * 60)
print("End of Model")
print("=" * 60)
