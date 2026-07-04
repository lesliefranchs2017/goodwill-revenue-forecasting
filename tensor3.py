# Wednesday, July 1, 2026

# import tensorflow as tf

# print("TensorFlow Version:", tf.__version__)

# x = tf.constant([1, 2, 3])

# print(x)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ]) 

import tensorflow as tf
import pandas as pd
import numpy as np
print("                                            ")
print("TensorFlow Version:", tf.__version__)

df = pd.read_csv("goodwill_mock_data_3000_revised_noincome.csv")
print("                                            ")
print("printing head")
print(df.head())
print("                                            ")
print("printing shape")
print(df.shape)
[print("printing columns")]
print(df.columns) 

print("                                            ")
print("print(df.info())")
print(df.info())
print("print(df.describe())")
print(print(df.describe()))

print("                                            ")
feature_cols = ['Expenses', 'Unemployment', 'CPI', 'BarrelOil', 'NetAssets']
target_col = 'Revenue'
print("                                            ")
X = df[feature_cols].values.astype('float32')
y = df[target_col].values.astype('float32')
print("                                            ")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("                                            ")
print("                                            ")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation="relu", input_shape=(5,)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Then save the file and run:
# python tensor5.py
# This is the point where TensorFlow starts learning the weights and bias from your 2,400 training rows.