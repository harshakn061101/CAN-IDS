import numpy as np
import torch
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from model import LSTMAutoencoder

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

app = FastAPI(title="CAN Intrusion Detection System")

# Allow the browser-based frontend (opened as a local file, or hosted anywhere)
# to call this API directly. Wide open (allow_origins=["*"]) is fine here since
# /predict and /health don't expose secrets or perform side effects — this is a
# read-only demo endpoint, not something that needs origin-restricted access.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WINDOW_SIZE       = 50
FREQ_THRESHOLD    = 0.664978
BYTEDEV_THRESHOLD = 0.85
RECON_THRESHOLD   = 0.043970

FEATURE_COLUMNS = ["can_id", "dlc", "d0","d1","d2","d3","d4","d5","d6","d7",
                    "freq", "inter_arrival", "byte_dev"]

scaler = joblib.load("src/scaler.pkl")
id_means = joblib.load("src/id_means.pkl")
id_stds  = joblib.load("src/id_stds.pkl")

lstm_model = LSTMAutoencoder(
    input_size=13,
    hidden_size=32,
    bottleneck_size=8,
    num_layers=2
)
lstm_model.load_state_dict(
    torch.load("src/lstm_autoencoder_best.pth", map_location="cpu")
)
lstm_model.eval()
print("Model and scaler loaded. Input size: 13 features.")

# Load reference distribution (from training data) once at startup
try:
    reference_data = pd.read_csv("src/reference_data.csv")
    print(f"Reference data loaded for drift monitoring: {reference_data.shape}")
except FileNotFoundError:
    reference_data = None
    print("WARNING: src/reference_data.csv not found. /drift-report will be unavailable.")

# Rolling buffer of recent production requests (last message of each /predict call)
PRODUCTION_LOG_MAXLEN = 5000
production_log = deque(maxlen=PRODUCTION_LOG_MAXLEN)


class CANMessage(BaseModel):
    can_id:        int
    dlc:           int
    d0: int; d1: int; d2: int; d3: int
    d4: int; d5: int; d6: int; d7: int
    freq:          float
    inter_arrival: float
    byte_dev:      float

class CANSequence(BaseModel):
    messages: List[CANMessage]

def messages_to_tensor(messages: List[CANMessage]):
    rows = []
    for m in messages:
        rows.append([
            m.can_id, m.dlc,
            m.d0, m.d1, m.d2, m.d3, m.d4, m.d5, m.d6, m.d7,
            m.freq, m.inter_arrival, m.byte_dev
        ])
    arr = np.array(rows, dtype=np.float64)
    arr_scaled = scaler.transform(arr)
    tensor = torch.FloatTensor(arr_scaled).unsqueeze(0)
    return tensor, arr_scaled

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        "CAN-IDS LSTM Autoencoder",
        "features":     13,
        "attack_types": ["DoS", "Fuzzy", "Gear", "RPM"]
    }

@app.post("/predict")
def predict(sequence: CANSequence):
    if len(sequence.messages) != WINDOW_SIZE:
        return {"error": f"Expected {WINDOW_SIZE} messages, got {len(sequence.messages)}"}

    tensor, arr_scaled = messages_to_tensor(sequence.messages)

    with torch.no_grad():
        reconstructed = lstm_model(tensor)
        recon_error = torch.mean((tensor - reconstructed) ** 2).item()

    last_freq    = float(arr_scaled[-1, 10])
    last_bytedev = float(arr_scaled[-1, 12])

    freq_alert    = last_freq    > FREQ_THRESHOLD
    bytedev_alert = last_bytedev > BYTEDEV_THRESHOLD
    recon_alert   = recon_error  > RECON_THRESHOLD
    is_attack     = bool(freq_alert or bytedev_alert or recon_alert)

    attack_type = "UNKNOWN"
    if freq_alert and not bytedev_alert:
        attack_type = "DoS or Fuzzy (rate-based)"
    elif bytedev_alert and not freq_alert:
        attack_type = "Gear or RPM spoofing (content-based)"
    elif freq_alert and bytedev_alert:
        attack_type = "Combined rate + content attack"

    # Log ALL messages in this window for drift monitoring (not just the last one) —
    # logging only the tail message starved the drift log to 1 sample per request,
    # making the comparison unreliable at normal traffic volumes.
    production_log.extend(arr_scaled.tolist())

    return {
        "prediction":           "ATTACK" if is_attack else "NORMAL",
        "is_attack":            is_attack,
        "attack_type":          attack_type if is_attack else "N/A",
        "freq_alert":           freq_alert,
        "bytedev_alert":        bytedev_alert,
        "recon_alert":          recon_alert,
        "last_freq_scaled":     round(last_freq, 6),
        "last_bytedev_scaled":  round(last_bytedev, 6),
        "reconstruction_error": round(recon_error, 6),
        "thresholds": {
            "freq":    FREQ_THRESHOLD,
            "bytedev": BYTEDEV_THRESHOLD,
            "recon":   RECON_THRESHOLD
        }
    }

MAX_EVAL_ROWS    = 200_000
MAX_EVAL_WINDOWS = 20_000

def _compute_rolling_frequency(df, window_seconds=0.1):
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

def _compute_inter_arrival(df):
    inter_arrival = np.zeros(len(df))
    timestamps = df["timestamp"].values
    can_ids = df["can_id"].values
    last_seen = {}
    for i in range(len(df)):
        cid = can_ids[i]
        t = timestamps[i]
        inter_arrival[i] = t - last_seen[cid] if cid in last_seen else 0.0
        last_seen[cid] = t
    return inter_arrival

def _compute_metrics(y_pred, y_true):
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "true_positives": TP, "false_positives": FP, "false_negatives": FN, "true_negatives": TN
    }

@app.post("/evaluate")
async def evaluate_dataset(file: UploadFile = File(...)):
    """
    Accepts a raw CAN log CSV (same format as this project's original DoS_dataset.csv:
    no header, columns = timestamp, can_id, dlc, d0..d7, flag — can_id/data bytes as hex
    strings, flag as 'R' (normal) or 'T' (attack)). Runs the exact same feature engineering
    the model was trained with, then scores every window against the model's fixed,
    already-learned thresholds — this is a genuine re-run of the evaluation methodology
    against whatever the caller uploads, not a simulation.
    """
    try:
        df = pd.read_csv(
            file.file, header=None,
            names=["timestamp", "can_id", "dlc", "d0","d1","d2","d3","d4","d5","d6","d7", "flag"]
        )
    except Exception as e:
        return {"error": f"Could not parse CSV: {e}"}

    if len(df) > MAX_EVAL_ROWS:
        return {"error": f"File has {len(df)} rows; the live evaluator caps at {MAX_EVAL_ROWS} rows to keep response times reasonable."}

    df = df.dropna()
    if len(df) < WINDOW_SIZE + 10:
        return {"error": f"Need at least {WINDOW_SIZE + 10} valid rows after cleaning; got {len(df)}."}

    hex_cols = ["can_id", "d0","d1","d2","d3","d4","d5","d6","d7"]
    try:
        for col in hex_cols:
            df[col] = df[col].apply(lambda x: int(str(x).strip(), 16))
    except Exception as e:
        return {"error": f"Could not parse hex values in can_id/data byte columns: {e}. Expected the same raw format as this project's original DoS_dataset.csv."}

    if not set(df["flag"].astype(str).str.strip().unique()).issubset({"R", "T"}):
        return {"error": "flag column must contain only 'R' (normal) or 'T' (attack)."}
    df["flag"] = df["flag"].astype(str).str.strip().map({"R": 0, "T": 1})

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["freq"] = np.log1p(_compute_rolling_frequency(df))
    df["inter_arrival"] = np.log1p(_compute_inter_arrival(df) * 1000)

    data_cols = ["d0","d1","d2","d3","d4","d5","d6","d7"]
    df_means = df[["can_id"]].join(id_means.add_suffix("_mean"), on="can_id")
    df_stds  = df[["can_id"]].join(id_stds.add_suffix("_std"),  on="can_id")
    for col in data_cols:
        df_means[col + "_mean"] = df_means[col + "_mean"].fillna(0)
        df_stds[col + "_std"]   = df_stds[col + "_std"].fillna(1.0)
    z_scores = pd.DataFrame()
    for col in data_cols:
        z_scores[col] = np.abs((df[col].values - df_means[col + "_mean"].values) / df_stds[col + "_std"].values)
    df["byte_dev"] = np.log1p(z_scores.mean(axis=1))

    X_raw = df[FEATURE_COLUMNS].values
    y_true_all = df["flag"].values
    X_scaled = scaler.transform(X_raw)

    n_possible = len(X_scaled) - WINDOW_SIZE
    if n_possible <= 0:
        return {"error": "Not enough rows to build a single 50-message window."}

    step = max(1, n_possible // MAX_EVAL_WINDOWS)
    starts = list(range(0, n_possible, step))[:MAX_EVAL_WINDOWS]

    X_windows     = np.array([X_scaled[s:s+WINDOW_SIZE] for s in starts])
    y_windows     = np.array([y_true_all[s+WINDOW_SIZE-1] for s in starts])
    last_freqs    = np.array([X_scaled[s+WINDOW_SIZE-1, 10] for s in starts])
    last_bytedevs = np.array([X_scaled[s+WINDOW_SIZE-1, 12] for s in starts])

    X_tensor = torch.FloatTensor(X_windows)
    with torch.no_grad():
        reconstructed = lstm_model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=(1, 2)).numpy()

    y_pred_recon    = (errors > RECON_THRESHOLD).astype(int)
    y_pred_freq     = (last_freqs > FREQ_THRESHOLD).astype(int)
    y_pred_bytedev  = (last_bytedevs > BYTEDEV_THRESHOLD).astype(int)
    y_pred_combined = ((errors > RECON_THRESHOLD) | (last_freqs > FREQ_THRESHOLD) | (last_bytedevs > BYTEDEV_THRESHOLD)).astype(int)

    return {
        "rows_in_file":       len(df),
        "windows_evaluated":  len(starts),
        "normal_windows":     int((y_windows == 0).sum()),
        "attack_windows":     int((y_windows == 1).sum()),
        "thresholds": {"freq": FREQ_THRESHOLD, "bytedev": BYTEDEV_THRESHOLD, "recon": RECON_THRESHOLD},
        "strategies": {
            "reconstruction_error": _compute_metrics(y_pred_recon, y_windows),
            "frequency":            _compute_metrics(y_pred_freq, y_windows),
            "byte_deviation":       _compute_metrics(y_pred_bytedev, y_windows),
            "combined":             _compute_metrics(y_pred_combined, y_windows),
        }
    }

@app.get("/drift-report")
def drift_report():
    if reference_data is None:
        return {"error": "Reference data not loaded. Cannot compute drift."}

    if len(production_log) < 500:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(production_log)} data points logged so far. Need at least 500 to compute a reliable drift report.",
            "requests_logged": len(production_log)
        }

    current_data = pd.DataFrame(list(production_log), columns=FEATURE_COLUMNS)

    report = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=0.2)])
    report.run(reference_data=reference_data, current_data=current_data)
    result = report.as_dict()

    # metrics[0] = DatasetDriftMetric (summary only)
    # metrics[1] = DataDriftTable (has per-column drift_by_columns)
    summary_metric = result["metrics"][0]["result"]
    table_metric   = result["metrics"][1]["result"]

    dataset_drift = summary_metric.get("dataset_drift", None)
    drift_share = summary_metric.get("drift_share", None)
    n_drifted = summary_metric.get("number_of_drifted_columns", None)
    n_features = summary_metric.get("number_of_columns", None)

    drifted_features = []
    per_column = table_metric.get("drift_by_columns", {})
    for col_name, col_result in per_column.items():
        if col_result.get("drift_detected"):
            drifted_features.append({
                "feature": col_name,
                "drift_score": round(float(col_result.get("drift_score", 0)), 6),
                "test_used": col_result.get("stattest_name", "unknown")
            })

    return {
        "requests_analyzed": len(production_log),
        "dataset_drift_detected": dataset_drift,
        "drift_share": drift_share,
        "num_drifted_features": n_drifted,
        "total_features": n_features,
        "drifted_features": drifted_features
    }
