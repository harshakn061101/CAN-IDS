import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque
import joblib
import os

columns = [
    "timestamp", "can_id", "dlc",
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
    "flag"
]

print("Loading dataset...")

def load_dataset(filepath):
    df = pd.read_csv(filepath, header=None, names=columns)
    df = df.dropna()
    hex_cols = ["can_id", "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
    for col in hex_cols:
        df[col] = df[col].apply(lambda x: int(str(x).strip(), 16))
    df["flag"] = df["flag"].map({"R": 0, "T": 1})
    return df

# Personal laptop: load DoS only to avoid memory error
# Office laptop: load all four and concat
df = load_dataset("data/DoS_dataset.csv")
print(f"Shape: {df.shape}  attacks: {df['flag'].sum()}")

df = df.sort_values("timestamp").reset_index(drop=True)

print("Computing Feature 1: Rolling frequency...")

def compute_rolling_frequency(df, window_seconds=0.1):
    freqs = np.zeros(len(df))
    timestamps = df["timestamp"].values
    can_ids = df["can_id"].values
    id_windows = {}
    for i in range(len(df)):
        cid = can_ids[i]
        t = timestamps[i]
        if cid not in id_windows:
            id_windows[cid] = deque()
        window = id_windows[cid]
        window.append(t)
        while window and (t - window[0]) > window_seconds:
            window.popleft()
        freqs[i] = len(window)
    return freqs

df["freq"] = compute_rolling_frequency(df, window_seconds=0.1)
df["freq"] = np.log1p(df["freq"])

print("Computing Feature 2: Inter-arrival time...")

def compute_inter_arrival(df):
    inter_arrival = np.zeros(len(df))
    timestamps = df["timestamp"].values
    can_ids = df["can_id"].values
    last_seen = {}
    for i in range(len(df)):
        cid = can_ids[i]
        t = timestamps[i]
        if cid in last_seen:
            inter_arrival[i] = t - last_seen[cid]
        else:
            inter_arrival[i] = 0.0
        last_seen[cid] = t
    return inter_arrival

df["inter_arrival"] = compute_inter_arrival(df)
df["inter_arrival"] = np.log1p(df["inter_arrival"] * 1000)

print("Computing Feature 3: Per-ID byte deviation (vectorized)...")

data_cols = ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]

normal_rows = df[df["flag"] == 0]
id_means = normal_rows.groupby("can_id")[data_cols].mean()
id_stds  = normal_rows.groupby("can_id")[data_cols].std().fillna(1.0).replace(0, 1.0)

df_means = df[["can_id"]].join(id_means.add_suffix("_mean"), on="can_id")
df_stds  = df[["can_id"]].join(id_stds.add_suffix("_std"),  on="can_id")

for col in data_cols:
    df_means[col + "_mean"] = df_means[col + "_mean"].fillna(0)
    df_stds[col  + "_std"]  = df_stds[col  + "_std"].fillna(1.0)

z_scores = pd.DataFrame()
for col in data_cols:
    z_scores[col] = np.abs(
        (df[col].values - df_means[col + "_mean"].values) /
        df_stds[col + "_std"].values
    )

df["byte_dev"] = z_scores.mean(axis=1)
df["byte_dev"] = np.log1p(df["byte_dev"])

print("\nFeature stats by flag (0=normal, 1=attack):")
print(df.groupby("flag")[["freq", "inter_arrival", "byte_dev"]].mean())

feature_cols = [
    "can_id", "dlc",
    "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
    "freq", "inter_arrival", "byte_dev"
]
X = df[feature_cols].values
y = df["flag"].values

print(f"\nFeatures shape: {X.shape}")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

os.makedirs("src", exist_ok=True)
joblib.dump(scaler,   "src/scaler.pkl")
joblib.dump(id_means, "src/id_means.pkl")
joblib.dump(id_stds,  "src/id_stds.pkl")

np.save("src/X_train.npy", X_train)
np.save("src/X_test.npy",  X_test)
np.save("src/y_train.npy", y_train)
np.save("src/y_test.npy",  y_test)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")
print("All files saved to src/")