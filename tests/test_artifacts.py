import joblib
import numpy as np


def test_scaler_loads_and_has_expected_feature_count():
    """The scaler must know about all 13 features, or transform() will silently
    misalign columns for any request."""
    scaler = joblib.load("src/scaler.pkl")
    assert scaler.n_features_in_ == 13


def test_reference_data_matches_feature_count():
    """The Evidently drift reference dataset must have the same 13 columns as
    what /predict actually logs, or drift comparisons would be comparing
    mismatched columns silently."""
    import pandas as pd
    reference = pd.read_csv("src/reference_data.csv")
    assert reference.shape[1] == 13


def test_scaler_transform_roundtrip_is_consistent():
    """A basic sanity check: transforming and inverse-transforming a row should
    return (approximately) the original values — catches a corrupted or
    mismatched scaler file."""
    scaler = joblib.load("src/scaler.pkl")
    sample = np.zeros((1, 13))
    sample[0, 0] = 500  # arbitrary can_id-like value within a plausible range
    scaled = scaler.transform(sample)
    restored = scaler.inverse_transform(scaled)
    assert np.allclose(sample, restored, atol=1e-6)
