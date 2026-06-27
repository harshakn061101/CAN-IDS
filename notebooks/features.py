import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load clean data
df = pd.read_csv("data/dos_clean.csv")
print("Loaded:", df.shape)

# Step 1 — Select only the feature columns (drop timestamp and flag)
feature_cols = ["can_id", "dlc", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
X = df[feature_cols].values
y = df["flag"].values

print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Step 2 — Normalize features to 0-1 range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print("\nBefore scaling — can_id min/max:", X[:, 0].min(), X[:, 0].max())
print("After scaling  — can_id min/max:", X_scaled[:, 0].min(), X_scaled[:, 0].max())

# Step 3 — Split BEFORE anything else
# We split the UNSCALED data first, then scale
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 4 — NOW scale, fitting ONLY on train
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)  # learns min/max from train only
X_test = scaler.transform(X_test_raw)         # applies same scale, no re-learning

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTrain flag counts:", np.unique(y_train, return_counts=True))
print("Test flag counts:", np.unique(y_test, return_counts=True))