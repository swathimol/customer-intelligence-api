import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # the "with" runs startup, so models load
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_features"] == 25


def test_predict_parity_with_notebook(client):
    payload = json.load(open(ROOT / "example_predict.json"))
    expected = json.load(open(ROOT / "expected.json"))
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["raw_score"] == pytest.approx(expected["raw_score"], abs=1e-3)
    assert body["repurchase_probability_90d"] == pytest.approx(expected["probability"], abs=1e-3)


def test_predict_rejects_missing_feature(client):
    payload = json.load(open(ROOT / "example_predict.json"))
    del payload["features"]["recency"]
    assert client.post("/predict", json=payload).status_code == 422


def test_segment_parity(client):
    payload = json.load(open(ROOT / "example_segment.json"))
    expected = json.load(open(ROOT / "expected.json"))
    r = client.post("/segment", json=payload)
    assert r.status_code == 200
    assert r.json()["segment"] == expected["segment"]


def test_segment_rejects_negative(client):
    bad = {"recency": -1, "frequency": 1, "monetary": 1, "avg_basket_value": 1, "tenure": 1}
    assert client.post("/segment", json=bad).status_code == 422


def test_recommend_known_user_is_hybrid(client):
    r = client.get("/recommend/0?n=5")
    assert r.status_code == 200
    assert r.json()["source"] == "hybrid"
    assert len(r.json()["items"]) == 5


def test_recommend_unknown_user_falls_back(client):
    r = client.get("/recommend/999999999?n=5")
    assert r.json()["source"] == "popular_fallback"