"""Application behavior tests; no model artifacts or GPU are needed."""
import sys
import sqlite3
from types import ModuleType
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "acos-backend"))
from acos_app import create_app


@pytest.fixture
def app(tmp_path):
    return create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "test.sqlite3", "SEED_DEMO": True})


@pytest.fixture
def client(app):
    return app.test_client()


def import_body(product_id="user-laptop"):
    return {"domain": "laptop", "records": [{
        "product_id": product_id, "product_name": "Imported laptop", "text": "The battery is good.",
        "quads": [{"aspect": "battery", "category": "BATTERY#OPERATION_PERFORMANCE",
                   "opinion": "good", "sentiment": "Positive"}]}]}


def test_health_and_synthetic_catalog(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json["status"] == "ok"
    assert health.json["analysis_mode"] == "heuristic"
    assert health.json["model_loaded"] is False
    meta = client.get("/api/meta").json
    assert meta["domains"] == ["laptop", "rest16"]
    assert meta["synthetic_demo_products"] == 6
    for domain in meta["domains"]:
        catalog = client.get("/api/products", query_string={"domain": domain}).json
        assert catalog["count"] == 3
        assert all(product["is_demo"] for product in catalog["products"])
        assert all(product["review_count"] == 2 for product in catalog["products"])


def test_analysis_is_input_dependent_and_domain_aware(client):
    positive = client.post("/api/analyze", json={"sentence": "The battery is amazing.", "domain": "laptop"}).json
    negative = client.post("/api/analyze", json={"sentence": "The screen is dull.", "domain": "laptop"}).json
    assert positive["results"] == [{"aspect": "battery", "category": "BATTERY#OPERATION_PERFORMANCE",
                                   "opinion": "amazing", "sentiment": "Positive"}]
    assert negative["results"] == [{"aspect": "screen", "category": "DISPLAY#QUALITY",
                                   "opinion": "dull", "sentiment": "Negative"}]
    assert positive["analysis_mode"] == "heuristic"
    assert "not pretrained" in positive["notice"]
    assert client.post("/api/analyze", json={"sentence": "No recognized feature here."}).json["results"] == []
    assert client.post("/api/analyze", json={"sentence": "The food is tasty.", "domain": "rest16"}).json["results"][0]["category"] == "FOOD#QUALITY"
    assert client.post("/api/analyze", json={"sentence": "The food is tasty.", "domain": "laptop"}).json["results"] == []


def test_clause_polarity_negation_and_korean(client):
    reply = client.post("/api/analyze", json={"sentence": "The battery is good but the screen is bad."}).json
    assert {item["aspect"]: item["sentiment"] for item in reply["results"]} == {"battery": "Positive", "screen": "Negative"}
    for sentence, expected in (("The battery is not good.", "Negative"),
                               ("The battery is not bad.", "Positive"),
                               ("배터리는 좋다.", "Positive"), ("배터리는 짧다.", "Negative")):
        assert client.post("/api/analyze", json={"sentence": sentence}).json["results"][0]["sentiment"] == expected


@pytest.mark.parametrize("body", [None, [], {}, {"sentence": 1}, {"sentence": True},
    {"sentence": " "}, {"sentence": "x" * 8001}, {"sentence": "good", "domain": []},
    {"sentence": "good", "domain": "clothing"}, {"sentence": "good", "path": "C:/secret"}])
def test_invalid_analysis_requests_are_clean_400(client, body):
    reply = client.post("/api/analyze", json=body) if body is not None else client.post("/api/analyze", data="null", content_type="application/json")
    assert reply.status_code == 400
    assert set(reply.json) == {"error", "code"}
    assert "Traceback" not in reply.get_data(as_text=True)


def test_json_media_and_body_limits(client):
    assert client.post("/api/analyze", data="sentence=test").status_code == 415
    assert client.post("/api/analyze", data="{oops", content_type="application/json").status_code == 400
    assert client.post("/api/analyze", data="x" * 2_000_001, content_type="application/json").status_code == 413


def test_seed_does_not_overwrite_or_duplicate_existing_data(tmp_path):
    path = tmp_path / "persistent.sqlite3"
    config = {"TESTING": True, "DATABASE_PATH": path, "SEED_DEMO": False}
    first = create_app(config).test_client()
    assert first.get("/api/products").json["count"] == 0
    assert first.post("/api/import", json=import_body()).status_code == 200
    second = create_app({**config, "SEED_DEMO": True}).test_client()
    assert second.get("/api/products").json["count"] == 1
    assert second.get("/api/meta").json["synthetic_demo_products"] == 0
    assert second.get("/api/products/user-laptop").json["product"]["is_demo"] is False


def test_seed_is_idempotent(app):
    second = create_app({"TESTING": True, "DATABASE_PATH": app.config["DATABASE_PATH"], "SEED_DEMO": True}).test_client()
    assert second.get("/api/products").json["count"] == 3
    assert second.get("/api/products/demo-laptop-aurora").json["product"]["quad_count"] == 4


def test_import_counts_idempotency_and_evidence(client):
    first = client.post("/api/import", json=import_body())
    assert first.status_code == 200
    assert first.json["imported"] == {"products": 1, "reviews": 1, "quads": 1, "records": 1, "skipped_records": 0}
    second = client.post("/api/import", json=import_body()).json
    assert second["imported"] == {"products": 0, "reviews": 0, "quads": 0, "records": 1, "skipped_records": 1}
    detail = client.get("/api/products/user-laptop").json
    assert detail["product"]["review_count"] == 1
    assert detail["reviews"][0]["quads"][0]["sentiment"] == "Positive"
    assert detail["features"][0]["score"] == 1


def test_import_validates_complete_batch_before_write(client):
    body = import_body()
    body["records"].append({**body["records"][0], "product_id": "bad", "quads": [{
        "aspect": "battery", "category": "BATTERY#OPERATION_PERFORMANCE", "opinion": "bad", "sentiment": 0}]})
    assert client.post("/api/import", json=body).status_code == 400
    assert client.get("/api/products/user-laptop").status_code == 404


def test_import_product_conflict_rolls_back_complete_transaction(client):
    body = import_body()
    body["records"].append({**body["records"][0], "product_id": "demo-laptop-aurora", "product_name": "Conflicting name"})
    assert client.post("/api/import", json=body).status_code == 409
    assert client.get("/api/products/user-laptop").status_code == 404
    assert client.get("/api/products/demo-laptop-aurora").json["product"]["name"] == "Aurora 14 (synthetic demo)"


def test_implicit_slots_and_empty_annotations_are_preserved(client):
    body = import_body()
    body["records"][0]["quads"][0].update(aspect="", opinion="")
    body["records"].append({**body["records"][0], "text": "Unannotated review", "quads": []})
    assert client.post("/api/import", json=body).status_code == 200
    detail = client.get("/api/products/user-laptop").json
    assert detail["product"]["review_count"] == 2
    assert detail["product"]["quad_count"] == 1
    assert detail["features"][0]["feature"] == "(implicit)"


@pytest.mark.parametrize("mutation", [
    lambda body: body.update(domain="unknown"),
    lambda body: body.update(records=[]),
    lambda body: body.update(records=[body["records"][0]] * 1001),
    lambda body: body.update(path="C:/arbitrary/file"),
    lambda body: body["records"][0].update(product_id="../traversal"),
    lambda body: body["records"][0].update(product_name=False),
    lambda body: body["records"][0].update(quads=[body["records"][0]["quads"][0]] * 65),
    lambda body: body["records"][0]["quads"][0].update(sentiment="positive"),
    lambda body: body["records"][0]["quads"][0].update(category="bad label"),
])
def test_strict_import_validation(client, mutation):
    body = import_body()
    mutation(body)
    assert client.post("/api/import", json=body).status_code == 400
    assert client.get("/api/products/user-laptop").status_code == 404


def test_recommendations_have_real_sorted_evidence(client):
    body = {"domain": "laptop", "selections": {"BATTERY#OPERATION_PERFORMANCE": "battery"}, "mode": "AND", "top_k": 10}
    items = client.post("/api/recommend", json=body).json["items"]
    assert len(items) == 3
    assert [item["score"] for item in items] == [1, 1, -1]
    assert all(-1 <= item["score"] <= 1 for item in items)
    assert all(item["evidence"] for item in items)
    assert items[-1]["evidence"][0]["sentiment"] == "Negative"
    body["selections"]["FANS_COOLING#QUALITY"] = "fan"
    and_items = client.post("/api/recommend", json=body).json["items"]
    assert [item["product_id"] for item in and_items] == ["demo-laptop-orbit"]
    body["mode"] = "OR"
    assert client.post("/api/recommend", json=body).json["count"] == 3
    body["domain"] = "rest16"
    assert client.post("/api/recommend", json=body).json["count"] == 0


@pytest.mark.parametrize("top_k", [True, 0, 51, 1.5, "10"])
def test_recommendation_integer_limit(client, top_k):
    assert client.post("/api/recommend", json={"domain": "laptop", "selections": {"DISPLAY#QUALITY": "screen"}, "top_k": top_k}).status_code == 400


def test_categories_match_product_annotations(client):
    categories = client.get("/api/categories?domain=laptop").json["categories"]
    battery = next(item for item in categories if item["category"] == "BATTERY#OPERATION_PERFORMANCE")["features"][0]
    assert battery == {"feature": "battery", "count": 3, "pos": 2, "neu": 0, "neg": 1, "score": 0.3333}
    assert client.get("/api/products?domain=unknown").status_code == 400
    assert client.get("/api/products?domain=laptop&domain=rest16").status_code == 400
    assert client.get("/api/products?path=C:/secret").status_code == 400
    assert client.get("/api/products/missing-id").status_code == 404


def test_model_mode_never_falls_back_to_heuristic(tmp_path):
    client = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "model.sqlite3", "SEED_DEMO": False, "ANALYSIS_MODE": "model"}).test_client()
    assert client.get("/api/health").json["model_loaded"] is False
    reply = client.post("/api/analyze", json={"sentence": "The battery is great."})
    assert reply.status_code == 503
    assert reply.json["code"] == "model_unavailable"
    assert "results" not in reply.json


def test_explicit_predictor_injection_validates_result(tmp_path):
    calls = []
    def predictor(sentence, domain):
        calls.append((sentence, domain))
        return [{"aspect": "battery", "category": "BATTERY#OPERATION_PERFORMANCE", "opinion": "good", "sentiment": "Positive"}]
    config = {"TESTING": True, "DATABASE_PATH": tmp_path / "injected.sqlite3", "SEED_DEMO": False,
              "ANALYSIS_MODE": "model", "PREDICTOR": predictor}
    client = create_app(config).test_client()
    assert client.get("/api/health").json["model_loaded"] is True
    assert client.post("/api/analyze", json={"sentence": "Sample", "domain": "rest16"}).status_code == 200
    assert calls == [("Sample", "rest16")]
    bad = create_app({**config, "PREDICTOR": lambda *_: [{"sentiment": object()}]}).test_client()
    reply = bad.post("/api/analyze", json={"sentence": "Sample"})
    assert reply.status_code == 500
    assert reply.json["code"] == "predictor_failed"
    assert "Traceback" not in reply.get_data(as_text=True)


def test_browser_cors_is_local_only(client):
    assert client.get("/api/health", headers={"Origin": "http://localhost:8501"}).headers["Access-Control-Allow-Origin"] == "http://localhost:8501"
    assert "Access-Control-Allow-Origin" not in client.get("/api/health", headers={"Origin": "https://untrusted.example"}).headers


def test_trusted_environment_predictor_factory(tmp_path, monkeypatch):
    module = ModuleType("acos_test_adapter")
    calls = []
    def factory():
        calls.append("loaded")
        return lambda sentence, domain: []
    module.build = factory
    monkeypatch.setitem(sys.modules, "acos_test_adapter", module)
    monkeypatch.setenv("ACOS_PREDICTOR_FACTORY", "acos_test_adapter:build")
    config = {"TESTING": True, "DATABASE_PATH": tmp_path / "factory.sqlite3", "SEED_DEMO": False}
    heuristic = create_app(config).test_client()
    assert not calls
    assert heuristic.get("/api/health").json["analysis_mode"] == "heuristic"
    model = create_app({**config, "ANALYSIS_MODE": "model"}).test_client()
    assert calls == ["loaded"]
    assert model.get("/api/health").json["model_loaded"] is True
    reply = model.post("/api/analyze", json={"sentence": "The battery is great."})
    assert reply.status_code == 200
    assert reply.json["analysis_mode"] == "model"
    assert reply.json["results"] == []


@pytest.mark.parametrize("reference", ["missing_private_module:build", "invalid/module:factory", "json:missing"])
def test_predictor_factory_failure_leaves_model_unavailable(tmp_path, reference):
    client = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "broken.sqlite3", "SEED_DEMO": False,
                         "ANALYSIS_MODE": "model", "PREDICTOR_FACTORY": reference}).test_client()
    assert client.get("/api/health").json["status"] == "degraded"
    assert client.post("/api/analyze", json={"sentence": "The battery is good."}).status_code == 503
    assert reference not in client.get("/api/health").get_data(as_text=True)


def test_custom_exact_cors_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("ACOS_ALLOWED_ORIGINS", "https://acos.example,http://localhost:8502")
    client = create_app({"TESTING": True, "DATABASE_PATH": tmp_path / "cors.sqlite3", "SEED_DEMO": False}).test_client()
    assert client.get("/api/health", headers={"Origin": "https://acos.example"}).headers["Access-Control-Allow-Origin"] == "https://acos.example"
    assert "Access-Control-Allow-Origin" not in client.get("/api/health", headers={"Origin": "https://evil.example"}).headers


@pytest.mark.parametrize("origin", ["*", "https://.*", "https://example.org/path", "https://user:secret@example.org",
                                   "https://example.org?path=secret", "file:///C:/private", "http://localhost:99999"])
def test_wildcard_and_invalid_cors_configuration_rejected(tmp_path, origin):
    with pytest.raises(ValueError):
        create_app({"DATABASE_PATH": tmp_path / "badcors.sqlite3", "SEED_DEMO": False, "ALLOWED_ORIGINS": origin})


def test_sql_and_html_text_are_stored_as_literals(client):
    body = import_body()
    body["records"][0].update(product_name="<script>alert('test')</script>",
                               text="'; DROP TABLE products; -- <script>alert('test')</script>")
    assert client.post("/api/import", json=body).status_code == 200
    detail = client.get("/api/products/user-laptop").json
    assert detail["product"]["name"] == body["records"][0]["product_name"]
    assert detail["reviews"][0]["text"] == body["records"][0]["text"]
    assert client.get("/api/products").json["count"] == 4


def test_incompatible_database_is_preserved(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE products(id TEXT PRIMARY KEY, name TEXT)")
        connection.execute("INSERT INTO products VALUES('legacy-id', 'Existing user data')")
    with pytest.raises(RuntimeError, match="incompatible schema"):
        create_app({"DATABASE_PATH": path, "SEED_DEMO": True})
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM products").fetchall() == [("legacy-id", "Existing user data")]
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("products",)]


def test_relative_database_path_resolves_against_project_root(tmp_path, monkeypatch):
    import acos_app
    monkeypatch.setattr(acos_app, "PROJECT_ROOT", tmp_path)
    app = create_app({"DATABASE_PATH": "data/relative.sqlite3", "SEED_DEMO": False})
    assert app.config["DATABASE_PATH"] == str(tmp_path / "data" / "relative.sqlite3")
    assert (tmp_path / "data" / "relative.sqlite3").is_file()


def test_existing_product_cannot_be_reassigned_to_another_domain(client):
    body = import_body()
    assert client.post("/api/import", json=body).status_code == 200
    body["domain"] = "rest16"
    assert client.post("/api/import", json=body).status_code == 409
    assert client.get("/api/products/user-laptop").json["product"]["domain"] == "laptop"
