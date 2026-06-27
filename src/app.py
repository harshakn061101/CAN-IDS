import numpy as np
import torch
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from model import LSTMAutoencoder

app = FastAPI(title="CAN Intrusion Detection System")

WINDOW_SIZE       = 50
FREQ_THRESHOLD    = 0.664978
BYTEDEV_THRESHOLD = 0.85
RECON_THRESHOLD   = 0.043970

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