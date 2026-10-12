"""The old fixed mock contract is replaced by an input-dependent API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "acos-backend"))
from acos_app import create_app


def test_legacy_analyze_alias_preserves_functional_contract(tmp_path):
    client = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "alias.sqlite3", "SEED_DEMO": False}).test_client()
    assert client.post("/analyze", json={}).status_code == 400
    first = client.post("/analyze", json={"sentence": "The battery is amazing."})
    second = client.post("/api/analyze", json={"sentence": "The battery is amazing."})
    assert first.status_code == second.status_code == 200
    assert first.json == second.json
    assert len(first.json["results"]) == 1
    assert first.json["analysis_mode"] == "heuristic"
    assert client.post("/analyze", json={"sentence": "Unrelated synthetic contract check"}).json["results"] == []
