import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.bottleneck = nn.Linear(hidden_size, bottleneck_size)
        self.expand = nn.Linear(bottleneck_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        encoded, _ = self.encoder(x)
        compressed = self.bottleneck(encoded)
        expanded = self.expand(compressed)
        decoded, _ = self.decoder(expanded)
        return decoded


if __name__ == "__main__":
    model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
    fake_input = torch.randn(4, 50, 13)
    output = model(fake_input)
    print("Input shape: ", fake_input.shape)
    print("Output shape:", output.shape)
    print(model)