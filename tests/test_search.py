import os
import tempfile
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import database
from src.metrics import compute_ndcg_at_k, compute_ap
from src.main import app


@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Sets up a temporary SQLite database path for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_db_path = Path(temp_dir) / "test_recommender.db"
    monkeypatch.setattr(database, "DB_PATH", temp_db_path)
    database.init_db()
    yield
    if temp_db_path.exists():
        temp_db_path.unlink()


def test_database_logging():
    # Test logging search
    results = [("1", 0.95), ("2", 0.85)]
    sid = database.log_search(
        user_id="u001",
        query_text="blue jacket",
        image_path="",
        search_type="vector_text",
        results=results
    )
    assert sid > 0

    # Test loading search history
    history = database.get_search_history()
    assert len(history) == 1
    assert history[0]["user_id"] == "u001"
    assert history[0]["query_text"] == "blue jacket"
    assert history[0]["search_type"] == "vector_text"

    # Test loading search results
    saved_results = database.get_search_results(sid)
    assert len(saved_results) == 2
    assert saved_results[0]["item_id"] == "1"
    assert saved_results[0]["score"] == 0.95

    # Test logging grade
    database.log_grade(
        user_id="u001",
        item_id="1",
        query_text="blue jacket",
        image_path="",
        rating=5
    )
    grades = database.get_all_grades()
    assert len(grades) == 1
    assert grades[0]["item_id"] == "1"
    assert grades[0]["rating"] == 5


def test_metrics_calculation():
    # Perfect ranking
    relevances = [5.0, 4.0, 3.0, 0.0]
    ndcg = compute_ndcg_at_k(relevances, k=3)
    assert ndcg == 1.0

    # Worse ranking
    relevances_worse = [0.0, 4.0, 5.0]
    ndcg_worse = compute_ndcg_at_k(relevances_worse, k=3)
    assert ndcg_worse < 1.0

    # Zero IDCG case
    assert compute_ndcg_at_k([0.0, 0.0], k=2) == 0.0

    # Average Precision
    binary_rels = [True, False, True, False]
    ap = compute_ap(binary_rels)
    # Precision at rank 1: 1/1 = 1.0 (relevant)
    # Precision at rank 2: 1/2 = 0.5 (not relevant)
    # Precision at rank 3: 2/3 = 0.666 (relevant)
    # Precision at rank 4: 2/4 = 0.5 (not relevant)
    # AP = (1.0 + 0.666) / 2 = 0.8333...
    assert abs(ap - 0.8333) < 0.001
    assert compute_ap([False, False]) == 0.0


def test_api_endpoints():
    client = TestClient(app)
    with client:
        # Test grading endpoint
        grade_payload = {
            "user_id": "u001",
            "item_id": "10",
            "rating": 4,
            "query_text": "red shoes",
            "image_path": ""
        }
        resp = client.post("/grade", data=grade_payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        # Check search-history
        resp = client.get("/search-history")
        assert resp.status_code == 200
        assert "history" in resp.json()

        # Check metrics endpoint
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "overall" in data
        assert "NDCG@10" in data["overall"]
        assert "MAP" in data["overall"]
