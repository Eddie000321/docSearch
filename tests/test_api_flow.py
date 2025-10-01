import importlib
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_client(tmp_path, monkeypatch) -> TestClient:
    data_dir = tmp_path / "store"
    monkeypatch.setenv("EVIDENCE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("EVIDENCE_USE_FAKE_EMBEDDINGS", "1")

    for module in list(sys.modules):
        if module.split(".")[0] in {"app", "config", "services"}:
            sys.modules.pop(module, None)

    app_module = importlib.import_module("app")
    client = TestClient(app_module.app)
    return client


def test_ingest_and_ask_flow(api_client: TestClient):
    sample_path = Path(__file__).parent / "data" / "sample.txt"
    with sample_path.open("rb") as handle:
        response = api_client.post(
            "/api/v1/ingest",
            files={"file": ("sample.txt", handle, "text/plain")},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["num_chunks"] > 0
    document_id = payload["document_id"]

    ask_response = api_client.post(
        "/api/v1/ask",
        json={"question": "What does the system return?", "k": 3},
    )
    assert ask_response.status_code == 200
    answer_payload = ask_response.json()
    assert answer_payload["evidence"], "Expected at least one evidence snippet"
    assert any(item["doc_id"] == document_id for item in answer_payload["evidence"])

    health_response = api_client.get("/api/v1/health")
    assert health_response.status_code == 200
    assert health_response.json()["vector_count"] >= payload["num_chunks"]
