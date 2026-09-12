import numpy as np
import torch
import joblib
import pandas as pd
from fastapi import FastAPI
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
