import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("This is the first look at the data")
print("                                 ")
print("                                      ")
print("making a change for GIT")

csv_path = '/mnt/c/python/project/goodwill_mock_data_3000_revised_noincome.csv'

# Load CSV
df = pd.read_csv(csv_path, encoding="utf-8-sig")
print("Look at file and hope for 14 rows")
print(df)

print("Get the number of rows")
print(len(df))

# Clean column names
df.columns = df.columns.str.strip()

print("Columns found:", df.columns.tolist())
print(df.head())

# Make sure required columns exist
required_cols = [
    'Revenue',
    'Expenses',
    'CPI',
    'Unemployment',
    'BarrelOil',
    'NetAssets'
]

missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(
        f"CSV must contain {required_cols}. Missing: {missing}. Found: {df.columns.tolist()}"
    )

print("All required columns exist.")

# Keep only required rows and force numeric conversion
df = df.copy()
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=required_cols)

# Input and target
feature_cols = [
    'Expenses',
    'CPI',
    'Unemployment',
    'BarrelOil',
    'NetAssets'
]
target_col = 'Revenue'

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32).reshape(-1, 1)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Check for invalid values
if not np.isfinite(X).all():
    raise ValueError("X contains NaN or infinite values.")
if not np.isfinite(y).all():
    raise ValueError("y contains NaN or infinite values.")

# Standardize X
X_mean = X.mean(axis=0, keepdims=True)
X_std = X.std(axis=0, keepdims=True)
X_std[X_std == 0] = 1.0
X_scaled = (X - X_mean) / X_std

# Standardize y
y_mean = y.mean(axis=0, keepdims=True)
y_std = y.std(axis=0, keepdims=True)
y_std[y_std == 0] = 1.0
y_scaled = (y - y_mean) / y_std

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Define model
model = nn.Linear(in_features=5, out_features=1)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTrain the model with 5000 epochs")

epochs = 5000
for epoch in range(epochs):
    model.train()

    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)

    if torch.isnan(loss):
        raise ValueError(f"Loss became NaN at epoch {epoch + 1}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}")

# Evaluate
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_tensor)
    y_pred = y_pred_scaled.numpy() * y_std + y_mean

    # R-squared using original scale
    y_tensor_original = torch.tensor(y, dtype=torch.float32)
    y_pred_original = torch.tensor(y_pred, dtype=torch.float32)

    ss_total = torch.sum((y_tensor_original - torch.mean(y_tensor_original)) ** 2)
    ss_residual = torch.sum((y_tensor_original - y_pred_original) ** 2)
    r_sq = 1 - (ss_residual / ss_total)

print(f"\nCoefficient of determination (R-squared): {r_sq.item():.4f}")

# Convert coefficients back to original scale
with torch.no_grad():
    weights_scaled = model.weight.detach().numpy()[0]
    bias_scaled = model.bias.detach().numpy()[0]

    # Transform coefficients back to original units
    y_std_scalar = float(y_std[0][0])
    y_mean_scalar = float(y_mean[0][0])

    original_weights = y_std_scalar * (weights_scaled / X_std.flatten())
    original_bias = y_mean_scalar + y_std_scalar * bias_scaled - np.sum(original_weights * X_mean.flatten())

print(f"Intercept: {original_bias:.5f}")
print("Slopes (Coefficients):")
for col, coef in zip(feature_cols, original_weights):
    print(f"  {col}: {coef:.5f}")

# Predict existing values
print("\nPredicted Revenue for existing rows:")
print(y_pred.flatten())

# Predict Revenue from new values
x_new = np.array([[70000000, 250.142, 5.3, 93, 43309235.5333]], dtype=np.float32)

# Scale new input
x_new_scaled = (x_new - X_mean) / X_std
x_new_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)

with torch.no_grad():
    y_new_pred_scaled = model(x_new_tensor).numpy()
    y_new_pred = y_new_pred_scaled * y_std + y_mean
    y_new_pred_value = float(y_new_pred[0][0])

print(
    f"\nPredicted Revenue for Expenses = {x_new[0][0]}, "
    f"CPI = {x_new[0][1]}, "
    f"Unemployment = {x_new[0][2]}, "
    f"BarrelOil = {x_new[0][3]}, "
    f"NetAssets = {x_new[0][4]}: "
    f"{y_new_pred_value:.2f}"
)

# Build output DataFrame
if 'Year' in df.columns:
    years = df['Year'].values
else:
    years = list(range(1, len(y_pred) + 1))

df_existing = pd.DataFrame({
    "Year": years,
    "Predicted_Revenue": y_pred.flatten(),
    "Supplied Expenses": X[:, 0],
    "CPI": X[:, 1],
    "Unemployment": X[:, 2],
    "Barrel Oil": X[:, 3],
    "NetAssets": X[:, 4]
})

df_new = pd.DataFrame({
    "Year": ["Future"],
    "Predicted_Revenue": [y_new_pred_value],
    "Supplied Expenses": [x_new[0][0]],
    "CPI": [x_new[0][1]],
    "Unemployment": [x_new[0][2]],
    "Barrel Oil": [x_new[0][3]],
    "NetAssets": [x_new[0][4]],
})

print("\nNew prediction row:")
print(df_new)

df_final = pd.concat([df_existing, df_new], ignore_index=True)

# Export to CSV
output_csv = "/mnt/c/python/project/csv/Revenue_predictions_3000_noincome.csv"
df_final.to_csv(output_csv, index=False)

print(f"\nSaved predictions to: {output_csv}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(range(len(y)), y.flatten(), label='Actual Revenue')
plt.plot(range(len(y_pred)), y_pred.flatten(), label='Predicted Revenue')
plt.xlabel("Row Index")
plt.ylabel("Revenue")
plt.title("Actual vs Predicted Revenue")
plt.legend()
plt.grid(True)
plt.show()


import statsmodels.api as sm

# Features
X_sm = df[['Expenses',
           'CPI',
           'Unemployment',
           'BarrelOil',
           'NetAssets']]

# Add intercept
X_sm = sm.add_constant(X_sm)

# Target
y_sm = df['Revenue']

# Fit model
model_sm = sm.OLS(y_sm, X_sm).fit()

# Full summary
print(model_sm.summary())
