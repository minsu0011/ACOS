"""Flask application factory and the public ACOS application API."""
import importlib
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from . import storage, validation as v
from .analysis import HEURISTIC_NOTICE, MODEL_NOTICE, heuristic_predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def environment_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.casefold() not in ("true", "false", "1", "0"):
        raise ValueError(f"{name} must be true, false, 1 or 0.")
    return value.casefold() in ("true", "1")


def allowed_origins(value):
    """Flask-CORS accepts regexes too; this app permits exact origins only."""
    origins = value.split(",") if isinstance(value, str) else value
    if not isinstance(origins, (list, tuple)) or len(origins) > 16:
        raise ValueError("ACOS_ALLOWED_ORIGINS must contain at most 16 exact HTTP(S) origins.")
    result = []
    for origin in origins:
        if not isinstance(origin, str):
            raise ValueError("ACOS_ALLOWED_ORIGINS must contain exact HTTP(S) origins.")
        origin = origin.strip()
        if not origin:
            continue
        parsed = urlsplit(origin)
        if (len(origin) > 200 or parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment
                or any(char in origin for char in "*\\{}?$^()")
                or any(char.isspace() for char in origin)):
            raise ValueError("ACOS_ALLOWED_ORIGINS must contain exact HTTP(S) origins, without paths or wildcards.")
        # Accessing .port also rejects malformed/out-of-range port numbers.
        if parsed.port == 0:
            raise ValueError("ACOS_ALLOWED_ORIGINS ports must be between 1 and 65535.")
        if "[" in origin or "]" in origin:
            # Flask-CORS could interpret bracketed IPv6 text as a regex.
            raise ValueError("ACOS_ALLOWED_ORIGINS must use explicit DNS names or IPv4 addresses.")
        result.append(origin)
    return result


def load_predictor(app):
    """Operator-selected Python entry point; never derived from request data."""
    if app.config["ANALYSIS_MODE"] != "model" or app.config["PREDICTOR"] is not None:
        return
    reference = app.config["PREDICTOR_FACTORY"]
    if not reference:
        return
    try:
        if not isinstance(reference, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", reference):
            raise ValueError("Invalid predictor factory reference.")
        module_name, factory_name = reference.split(":")
        factory = getattr(importlib.import_module(module_name), factory_name)
        predictor = factory()
        predict = getattr(predictor, "predict", predictor)
        if not callable(predict):
            raise TypeError("Predictor must be callable or expose predict.")
        app.config["PREDICTOR"] = predictor
    except Exception as error:
        # No fallback and no filesystem paths or exception messages in the API.
        app.logger.error("Configured predictor factory could not load (%s).", type(error).__name__)


def create_app(config=None):
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE_PATH=os.environ.get("ACOS_DATABASE_PATH", str(PROJECT_ROOT / "data" / "acos.sqlite3")),
        SEED_DEMO=environment_bool("ACOS_SEED_DEMO", True),
        ANALYSIS_MODE=os.environ.get("ACOS_ANALYSIS_MODE", "heuristic"),
        PREDICTOR=None, PREDICTOR_FACTORY=os.environ.get("ACOS_PREDICTOR_FACTORY", ""),
        ALLOWED_ORIGINS=os.environ.get("ACOS_ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"),
        MAX_CONTENT_LENGTH=2_000_000,
    )
    if config:
        app.config.update(config)
    if app.config["ANALYSIS_MODE"] not in ("heuristic", "model"):
        raise ValueError("ACOS_ANALYSIS_MODE must be heuristic or model.")
    if not isinstance(app.config["SEED_DEMO"], bool):
        raise ValueError("SEED_DEMO must be a boolean.")
    database_path = Path(app.config["DATABASE_PATH"]).expanduser()
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    app.config["DATABASE_PATH"] = str(database_path)
    app.json.ensure_ascii = False
    # The dashboard uses server-to-server requests. Browser access is local only.
    origins = allowed_origins(app.config["ALLOWED_ORIGINS"])
    CORS(app, resources={r"/api/*": {"origins": origins}, r"/analyze": {"origins": origins}})
    app.teardown_appcontext(storage.close_db)
    storage.initialize(app)
    load_predictor(app)

    def mode_info():
        mode = app.config["ANALYSIS_MODE"]
        return {"analysis_mode": mode, "model_loaded": mode == "model" and app.config["PREDICTOR"] is not None,
                "notice": HEURISTIC_NOTICE if mode == "heuristic" else MODEL_NOTICE}

    def json_body():
        if not request.is_json:
            raise v.ApiError("Content-Type must be application/json.", 415, "unsupported_media_type")
        body = request.get_json(silent=True)
        if body is None:
            raise v.ApiError("Request body must contain valid JSON.")
        return body

    def query_domain():
        if set(request.args) - {"domain"} or len(request.args.getlist("domain")) > 1:
            raise v.ApiError("Only one domain query parameter is supported.")
        return v.domain(request.args.get("domain", "laptop"))

    @app.errorhandler(v.ApiError)
    def api_error(error):
        return jsonify(error=error.message, code=error.code), error.status

    @app.errorhandler(HTTPException)
    def http_error(error):
        messages = {400: "Malformed request.", 404: "Endpoint was not found.",
                    405: "HTTP method is not supported.", 413: "Request body exceeds the 2 MB limit.",
                    415: "Content-Type must be application/json."}
        return jsonify(error=messages.get(error.code, "Request could not be processed."),
                       code="http_error"), error.code

    @app.errorhandler(sqlite3.Error)
    def database_error(error):
        app.logger.error("ACOS database operation failed (%s).", type(error).__name__)
        return jsonify(error="Database is temporarily unavailable.", code="database_unavailable"), 503

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.error("ACOS request failed (%s).", type(error).__name__)
        return jsonify(error="Request could not be completed.", code="internal_error"), 500

    @app.get("/api/health")
    def health():
        storage.get_db().execute("SELECT 1")
        info = mode_info()
        status = "degraded" if info["analysis_mode"] == "model" and not info["model_loaded"] else "ok"
        return jsonify(status=status, **info)

    @app.get("/api/meta")
    def meta():
        count = storage.get_db().execute("SELECT COUNT(*) FROM products WHERE is_demo = 1").fetchone()[0]
        return jsonify(domains=list(v.DOMAINS), synthetic_demo_products=count,
                       demo_notice="Seed examples are synthetic, manually annotated reviews.",
                       limits={"body_bytes": app.config["MAX_CONTENT_LENGTH"], "sentence_chars": v.MAX_TEXT,
                               "import_records": v.MAX_RECORDS, "quads_per_review": v.MAX_QUADS}, **mode_info())

    @app.post("/api/analyze")
    @app.post("/analyze")
    def analyze():
        body = v.object_fields(json_body(), ("sentence", "domain"), ("sentence",))
        sentence = v.text(body["sentence"], "sentence", v.MAX_TEXT)
        selected_domain = v.domain(body.get("domain", "laptop"))
        if app.config["ANALYSIS_MODE"] == "heuristic":
            results = heuristic_predict(sentence, selected_domain)
        else:
            predictor = app.config["PREDICTOR"]
            if predictor is None:
                raise v.ApiError("No ACOS model predictor is configured. Install verified model artifacts and configure a predictor, or explicitly select heuristic mode.",
                                 503, "model_unavailable")
            try:
                # Callable or object.predict: signature (sentence, domain).
                predict = predictor.predict if hasattr(predictor, "predict") else predictor
                results = v.quads(predict(sentence, selected_domain), "predictor results")
            except Exception as error:
                app.logger.error("Configured predictor failed (%s).", type(error).__name__)
                raise v.ApiError("Configured model predictor failed or returned invalid quadruples.",
                                 500, "predictor_failed") from None
        info = mode_info()
        return jsonify(input_sentence=sentence, domain=selected_domain, results=results,
                       analysis_mode=info["analysis_mode"], notice=info["notice"])

    @app.get("/api/products")
    def products():
        values = storage.products(query_domain())
        return jsonify(products=values, count=len(values))

    @app.get("/api/products/<selected_product>")
    def product_detail(selected_product):
        return jsonify(storage.product_detail(v.product_id(selected_product)))

    @app.get("/api/categories")
    def categories():
        grouped = defaultdict(list)
        for feature in storage.aggregate_features(selected_domain=query_domain()):
            selected_category = feature.pop("category")
            grouped[selected_category].append(feature)
        values = [{"category": category, "features": features} for category, features in grouped.items()]
        return jsonify(categories=values, count=len(values))

    @app.post("/api/recommend")
    def recommend():
        body = v.object_fields(json_body(), ("domain", "selections", "mode", "top_k"), ("domain", "selections"))
        selected_domain = v.domain(body["domain"])
        selections = body["selections"]
        if not isinstance(selections, dict) or not 1 <= len(selections) <= 16:
            raise v.ApiError("selections must contain 1 to 16 category-to-feature choices.")
        selections = {v.category(category): v.text(feature, "feature") for category, feature in selections.items()}
        mode = body.get("mode", "AND")
        if not isinstance(mode, str) or mode not in ("AND", "OR"):
            raise v.ApiError("mode must be AND or OR.")
        top_k = body.get("top_k", 10)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 50:
            raise v.ApiError("top_k must be an integer from 1 to 50.")
        items = storage.recommendations(selected_domain, selections, mode, top_k)
        return jsonify(items=items, count=len(items), score_notice="Mean polarity of matching quadruples: (positive - negative) / total, range -1 to 1. Matching includes all sentiments.")

    @app.post("/api/import")
    def import_reviews():
        selected_domain, records = v.import_request(json_body())
        imported = storage.import_records(selected_domain, records)
        return jsonify(imported=imported, domain=selected_domain,
                       notice="Validated annotations stored without model inference. Repeated records are deduplicated.")

    return app
