import numpy as np

# Load the processed data
X_train = np.load("src/X_train.npy")
y_train = np.load("src/y_train.npy")
X_test = np.load("src/X_test.npy")
y_test = np.load("src/y_test.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Step 1 — Keep only NORMAL rows from training data
# The model must never see attack data during training
normal_mask = y_train == 0
X_train_normal = X_train[normal_mask]

print("\nNormal-only training rows:", X_train_normal.shape)
print("(Original training rows were:", X_train.shape[0], ")")

# Step 2 — Build sliding window sequences
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size]
        sequences.append(seq)
    return np.array(sequences)

WINDOW_SIZE = 50

# This will take a moment - we're building millions of sequences
print(f"\nBuilding sequences with window size {WINDOW_SIZE}...")
X_train_seq = create_sequences(X_train_normal, WINDOW_SIZE)

print("Sequence data shape:", X_train_seq.shape)