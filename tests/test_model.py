import torch
from model import LSTMAutoencoder


def test_output_shape_matches_input_shape():
    """The autoencoder must reconstruct sequences of the exact same shape it was given —
    this is the core assumption the reconstruction-error attack detection relies on."""
    model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
    fake_input = torch.randn(4, 50, 13)  # batch=4, window=50, features=13
    output = model(fake_input)
    assert output.shape == fake_input.shape


def test_output_shape_with_different_batch_size():
    """Batch size should be flexible (training uses 64, single predictions use 1)."""
    model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
    fake_input = torch.randn(1, 50, 13)
    output = model(fake_input)
    assert output.shape == (1, 50, 13)


def test_bottleneck_compresses_dimension():
    """Sanity check that the bottleneck layer is actually smaller than hidden_size —
    otherwise this isn't really acting as a compressive autoencoder."""
    model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
    assert model.bottleneck.out_features < model.bottleneck.in_features
    assert model.bottleneck.out_features == 8


def test_model_loads_saved_weights_without_error():
    """Confirms the actual trained checkpoint file is compatible with the current
    LSTMAutoencoder architecture definition — catches accidental architecture/checkpoint
    mismatches (e.g. someone changes hidden_size without retraining)."""
    model = LSTMAutoencoder(input_size=13, hidden_size=32, bottleneck_size=8, num_layers=2)
    state_dict = torch.load("src/lstm_autoencoder_best.pth", map_location="cpu")
    model.load_state_dict(state_dict)  # raises if shapes mismatch
    model.eval()
