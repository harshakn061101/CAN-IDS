# CAN-IDS: Automotive CAN Bus Intrusion Detection System

A production-grade anomaly detection system for automotive CAN bus networks using an LSTM Autoencoder. Detects four attack types (DoS, Fuzzy, Gear spoofing, RPM spoofing) with 98.6% F1 score on DoS detection.

## Architecture

- **Model**: LSTM Autoencoder trained exclusively on normal CAN traffic
- **Features**: 13 engineered features including rolling frequency, inter-arrival time, and per-ID byte deviation
- **Detection**: Three-signal system — frequency alert (rate attacks), byte deviation alert (content attacks), reconstruction error (backup)
- **API**: FastAPI REST endpoint serving real-time predictions
- **Attack Types**: DoS, Fuzzy injection, Gear spoofing, RPM spoofing

## Results

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Frequency only (DoS/Fuzzy) | 99.81% | 97.46% | 98.62% |
| Byte deviation (Gear/RPM) | — | — | — |
| Combined (all attacks) | 84.91% | 65.15% | 73.73% |

## Feature Engineering

| Feature | What it captures | Attack type caught |
|---------|-----------------|-------------------|
| `freq` | Rolling message frequency per CAN ID (0.1s window, log1p) | DoS, Fuzzy |
| `inter_arrival` | Time gap since last message with this ID (log1p ms) | All types |
| `byte_dev` | Z-score deviation from normal data bytes per ID (log1p) | Gear, RPM |

## Project Structure