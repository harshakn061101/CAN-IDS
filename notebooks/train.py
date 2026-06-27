import numpy as np
import torch
import torch.nn as nn
from model import LSTMAutoencoder
import time

print("Loading data...")
X_train = np.load("src/X_train.npy")
y_train = np.load("src/y_train.npy")

normal_mask = y_train == 0
X_train_normal = X_train[normal_mask]
print("Normal-only training rows:", X_train_normal.shape)

SUBSET_SIZE = 300000
X_train_normal = X_train_normal[:SUBSET_SIZE]
print("Subsetted to:", X_train_normal.shape)

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences)

WINDOW_SIZE = 50
print("Building sequences...")
X_seq = create_sequences(X_train_normal, WINDOW_SIZE)
print("Sequences shape:", X_seq.shape)

X_tensor = torch.FloatTensor(X_seq)

val_split = int(0.9 * len(X_tensor))
X_train_t = X_tensor[:val_split]
X_val_t   = X_tensor[val_split:]
print("Train sequences:", X_train_t.shape)
print("Val sequences:  ", X_val_t.shape)

model = LSTMAutoencoder(
    input_size=13,
    hidden_size=32,
    bottleneck_size=8,
    num_layers=2
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn   = nn.MSELoss()

EPOCHS     = 6
BATCH_SIZE = 64
best_val_loss = float("inf")

print("\nStarting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    num_batches = 0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        batch = X_train_t[i : i + BATCH_SIZE]
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss  += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches

    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        val_loss   = loss_fn(val_output, X_val_t).item()

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s")

    torch.save(model.state_dict(), "src/lstm_autoencoder.pth")
    print(f"  Checkpoint saved (epoch {epoch+1})")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "src/lstm_autoencoder_best.pth")
        print(f"  New best model saved (val loss: {val_loss:.6f})")

print("\nTraining complete.")
print(f"Best model: src/lstm_autoencoder_best.pth (val loss: {best_val_loss:.6f})")