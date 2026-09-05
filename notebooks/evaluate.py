import numpy as np
import torch
import torch.nn as nn
from model import LSTMAutoencoder
import mlflow
import os

model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
model.load_state_dict(torch.load("src/lstm_autoencoder_best.pth"))
model.eval()
print("Model loaded.")

X_test = np.load("src/X_test.npy")
y_test = np.load("src/y_test.npy")
print("Test data shape:", X_test.shape)
print("Test attacks:", y_test.sum(), "Normal:", (y_test==0).sum())

WINDOW_SIZE   = 50
NUM_SEQUENCES = 20000

np.random.seed(42)
max_start     = len(X_test) - WINDOW_SIZE
random_starts = np.random.randint(0, max_start, size=NUM_SEQUENCES)

def build_sequences_from_starts(data, labels, starts, window_size):
    sequences     = []
    seq_labels    = []
    last_freqs    = []
    last_bytedevs = []
    for i in starts:
        seq = data[i : i + window_size]
        label        = labels[i + window_size - 1]
        last_freq    = data[i + window_size - 1, 10]
        last_bytedev = data[i + window_size - 1, 12]
        sequences.append(seq)
        seq_labels.append(label)
        last_freqs.append(last_freq)
        last_bytedevs.append(last_bytedev)
    return (np.array(sequences), np.array(seq_labels),
            np.array(last_freqs), np.array(last_bytedevs))

print("\nBuilding test sequences...")
X_test_seq, y_test_seq, last_freqs, last_bytedevs = build_sequences_from_starts(
    X_test, y_test, random_starts, WINDOW_SIZE
)
print("Normal:", (y_test_seq==0).sum(), "Attack:", (y_test_seq==1).sum())

X_test_tensor = torch.FloatTensor(X_test_seq)

print("Calculating reconstruction errors...")
with torch.no_grad():
    reconstructed = model(X_test_tensor)
    errors = torch.mean(
        (X_test_tensor - reconstructed) ** 2, dim=(1, 2)
    ).numpy()

normal_errors   = errors[y_test_seq == 0]
normal_freqs    = last_freqs[y_test_seq == 0]
normal_bytedevs = last_bytedevs[y_test_seq == 0]

recon_threshold   = np.percentile(normal_errors,   99)
freq_threshold    = np.percentile(normal_freqs,    99)
bytedev_threshold = np.percentile(normal_bytedevs, 99)

print(f"\nRecon threshold:   {recon_threshold:.6f}")
print(f"Freq threshold:    {freq_threshold:.6f}")
print(f"ByteDev threshold: {bytedev_threshold:.6f}")

def compute_metrics(y_pred, y_true):
    TP = ((y_pred==1) & (y_true==1)).sum()
    FP = ((y_pred==1) & (y_true==0)).sum()
    FN = ((y_pred==0) & (y_true==1)).sum()
    TN = ((y_pred==0) & (y_true==0)).sum()
    precision = TP/(TP+FP) if (TP+FP) > 0 else 0
    recall    = TP/(TP+FN) if (TP+FN) > 0 else 0
    f1        = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, f1, TP, FP, FN, TN

def print_metrics(precision, recall, f1, TP, FP, FN, TN, label):
    print(f"\n{label}")
    print(f"  TP:{TP}  FP:{FP}  FN:{FN}  TN:{TN}")
    print(f"  Precision:{precision:.4f}  Recall:{recall:.4f}  F1:{f1:.4f}")

y_pred_recon    = (errors > recon_threshold).astype(int)
y_pred_freq     = (last_freqs > freq_threshold).astype(int)
y_pred_bytedev  = (last_bytedevs > bytedev_threshold).astype(int)
y_pred_combined = (
    (errors > recon_threshold) |
    (last_freqs > freq_threshold) |
    (last_bytedevs > bytedev_threshold)
).astype(int)

metrics_recon    = compute_metrics(y_pred_recon,    y_test_seq)
metrics_freq     = compute_metrics(y_pred_freq,     y_test_seq)
metrics_bytedev  = compute_metrics(y_pred_bytedev,  y_test_seq)
metrics_combined = compute_metrics(y_pred_combined, y_test_seq)

print_metrics(*metrics_recon,    "Strategy A — Reconstruction error only")
print_metrics(*metrics_freq,     "Strategy B — Frequency only")
print_metrics(*metrics_bytedev,  "Strategy C — Byte deviation only")
print_metrics(*metrics_combined, "Strategy D — Combined (OR of all three)")

# --- MLflow logging: resume the same run that train.py started ---
run_id_path = "src/mlflow_run_id.txt"
if os.path.exists(run_id_path):
    with open(run_id_path) as f:
        run_id = f.read().strip()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("recon_threshold",   float(recon_threshold))
        mlflow.log_param("freq_threshold",    float(freq_threshold))
        mlflow.log_param("bytedev_threshold", float(bytedev_threshold))

        mlflow.log_metric("recon_precision", metrics_recon[0])
        mlflow.log_metric("recon_recall",    metrics_recon[1])
        mlflow.log_metric("recon_f1",        metrics_recon[2])

        mlflow.log_metric("freq_precision", metrics_freq[0])
        mlflow.log_metric("freq_recall",    metrics_freq[1])
        mlflow.log_metric("freq_f1",        metrics_freq[2])

        mlflow.log_metric("bytedev_precision", metrics_bytedev[0])
        mlflow.log_metric("bytedev_recall",    metrics_bytedev[1])
        mlflow.log_metric("bytedev_f1",        metrics_bytedev[2])

        mlflow.log_metric("combined_precision", metrics_combined[0])
        mlflow.log_metric("combined_recall",    metrics_combined[1])
        mlflow.log_metric("combined_f1",        metrics_combined[2])

    print(f"\nEvaluation metrics logged to MLflow run: {run_id}")
else:
    print("\nWARNING: src/mlflow_run_id.txt not found — run train.py first so evaluate.py can log into the same MLflow run.")
