import numpy as np
import torch
import torch.nn as nn
from model import LSTMAutoencoder
import matplotlib.pyplot as plt

model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
model.load_state_dict(torch.load("src/lstm_autoencoder_best.pth"))
model.eval()
print("Model loaded.")

X_test = np.load("src/X_test.npy")
y_test = np.load("src/y_test.npy")

WINDOW_SIZE   = 50
NUM_SEQUENCES = 20000

np.random.seed(42)
max_start     = len(X_test) - WINDOW_SIZE
random_starts = np.random.randint(0, max_start, size=NUM_SEQUENCES)

def build_sequences_from_starts(data, labels, starts, window_size):
    sequences  = []
    seq_labels = []
    for i in starts:
        seq   = data[i : i + window_size]
        label = labels[i + window_size - 1]
        sequences.append(seq)
        seq_labels.append(label)
    return np.array(sequences), np.array(seq_labels)

X_test_seq, y_test_seq = build_sequences_from_starts(X_test, y_test, random_starts, WINDOW_SIZE)
print("Normal:", (y_test_seq==0).sum(), "Attack:", (y_test_seq==1).sum())

X_test_tensor = torch.FloatTensor(X_test_seq)

with torch.no_grad():
    reconstructed = model(X_test_tensor)
    errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

normal_errors = errors[y_test_seq == 0]
attack_errors = errors[y_test_seq == 1]

print("\n--- NORMAL ---")
print(f"Mean: {normal_errors.mean():.6f}  Std: {normal_errors.std():.6f}")

print("\n--- ATTACK ---")
print(f"Mean: {attack_errors.mean():.6f}  Std: {attack_errors.std():.6f}")

freq_errors = torch.mean(
    (X_test_tensor[:, :, 10] - reconstructed[:, :, 10]) ** 2, dim=1
).numpy()
print(f"\nFreq-only recon — Normal: {freq_errors[y_test_seq==0].mean():.6f}  Attack: {freq_errors[y_test_seq==1].mean():.6f}")

bytedev_errors = torch.mean(
    (X_test_tensor[:, :, 12] - reconstructed[:, :, 12]) ** 2, dim=1
).numpy()
print(f"ByteDev-only recon — Normal: {bytedev_errors[y_test_seq==0].mean():.6f}  Attack: {bytedev_errors[y_test_seq==1].mean():.6f}")

plt.figure(figsize=(10, 6))
plt.hist(normal_errors, bins=50, alpha=0.6, label=f"Normal (n={len(normal_errors)})", color="green")
plt.hist(attack_errors, bins=50, alpha=0.6, label=f"Attack (n={len(attack_errors)})", color="red")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution: Normal vs Attack")
plt.legend()
plt.savefig("src/error_distribution.png")
print("\nPlot saved to src/error_distribution.png")
plt.show()