import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def _make_dummy_message(can_id=100, dlc=8, byte_val=0, freq=0.3, inter_arrival=0.2, byte_dev=0.1):
    return {
        "can_id": can_id, "dlc": dlc,
        "d0": byte_val, "d1": byte_val, "d2": byte_val, "d3": byte_val,
        "d4": byte_val, "d5": byte_val, "d6": byte_val, "d7": byte_val,
        "freq": freq, "inter_arrival": inter_arrival, "byte_dev": byte_dev
    }


def test_health_returns_expected_shape():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["features"] == 13
    assert set(body["attack_types"]) == {"DoS", "Fuzzy", "Gear", "RPM"}


def test_predict_rejects_wrong_window_length():
    """Sending fewer than 50 messages should return a clear error, not crash the server."""
    short_sequence = {"messages": [_make_dummy_message() for _ in range(10)]}
    response = client.post("/predict", json=short_sequence)
    assert response.status_code == 200  # endpoint returns a 200 with an "error" field, not a 4xx
    body = response.json()
    assert "error" in body
    assert "50" in body["error"]


def test_predict_accepts_valid_window_and_returns_expected_fields():
    """A properly-formed 50-message window should return a complete prediction result."""
    sequence = {"messages": [_make_dummy_message() for _ in range(50)]}
    response = client.post("/predict", json=sequence)
    assert response.status_code == 200
    body = response.json()

    for key in ["prediction", "is_attack", "attack_type", "freq_alert",
                "bytedev_alert", "recon_alert", "reconstruction_error", "thresholds"]:
        assert key in body

    assert body["prediction"] in ["NORMAL", "ATTACK"]
    assert isinstance(body["is_attack"], bool)
    assert set(body["thresholds"].keys()) == {"freq", "bytedev", "recon"}


def test_predict_rejects_malformed_message_fields():
    """Missing a required field (e.g. can_id) should trigger FastAPI's validation error,
    not an unhandled exception."""
    bad_message = _make_dummy_message()
    del bad_message["can_id"]
    sequence = {"messages": [bad_message] * 50}
    response = client.post("/predict", json=sequence)
    assert response.status_code == 422  # FastAPI/Pydantic validation error
