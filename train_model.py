import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import joblib

# Parameters
seq_len = 30

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/sample.csv")

# Filter a single commodity (e.g., 'onion')
df = df[df["commodity"] == "onion"]

# Sort by date
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Extract and scale price data
prices = df["price"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

# Check if we have enough data
if len(scaled) <= seq_len:
    raise ValueError(f"Not enough data. Need more than {seq_len} rows after filtering.")

# Create sequences
X = []
y = []
for i in range(seq_len, len(scaled)):
    X.append(scaled[i - seq_len:i])
    y.append(scaled[i])

X = np.array(X)
y = np.array(y)

# Reshape X for RNN input (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Build the RNN model
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
from tensorflow.keras.losses import MeanSquaredError
model.compile(optimizer='adam', loss=MeanSquaredError())


# Train the model
model.fit(X, y, epochs=20, batch_size=4)

# Save model and scaler
model.save("model/rnn_model.h5")
joblib.dump(scaler, "model/scaler.save")

print("✅ Model training complete and saved to 'model/' folder.")
