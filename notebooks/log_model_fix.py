import torch
import mlflow
import mlflow.pytorch
from model import LSTMAutoencoder

# Load the already-trained best model
model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
model.load_state_dict(torch.load("src/lstm_autoencoder_best.pth"))
model.eval()
print("Best model loaded from src/lstm_autoencoder_best.pth")

# Resume the same MLflow run that train.py started (it failed only on the log_model line)
with open("src/mlflow_run_id.txt") as f:
    run_id = f.read().strip()

with mlflow.start_run(run_id=run_id):
    mlflow.pytorch.log_model(model, "lstm_autoencoder", serialization_format="pickle")
    print(f"Model artifact successfully logged to MLflow run: {run_id}")
