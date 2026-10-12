"""Transactional SQLite storage, with no reset or delete workflow."""
import sqlite3
from collections import defaultdict

from flask import current_app, g

from .demo import DEMO_RECORDS
from .validation import ApiError

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY, name TEXT NOT NULL, domain TEXT NOT NULL,
    is_demo INTEGER NOT NULL DEFAULT 0 CHECK(is_demo IN (0, 1))
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL REFERENCES products(product_id), text TEXT NOT NULL,
    UNIQUE(product_id, text)
);
CREATE TABLE IF NOT EXISTS quads (
    quad_id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES reviews(review_id),
    aspect TEXT NOT NULL, category TEXT NOT NULL, opinion TEXT NOT NULL,
    sentiment TEXT NOT NULL CHECK(sentiment IN ('Positive', 'Neutral', 'Negative')),
    feature TEXT NOT NULL,
    UNIQUE(review_id, aspect, category, opinion, sentiment)
);
CREATE INDEX IF NOT EXISTS idx_products_domain ON products(domain);
CREATE INDEX IF NOT EXISTS idx_reviews_product ON reviews(product_id);
CREATE INDEX IF NOT EXISTS idx_quads_feature ON quads(category, feature);
"""
REQUIRED_COLUMNS = {
    "products": {"product_id", "name", "domain", "is_demo"},
    "reviews": {"review_id", "product_id", "text"},
    "quads": {"quad_id", "review_id", "aspect", "category", "opinion", "sentiment", "feature"},
}


def connect(path):
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def get_db():
    if "database" not in g:
        g.database = connect(current_app.config["DATABASE_PATH"])
    return g.database


def close_db(_error=None):
    connection = g.pop("database", None)
    if connection is not None:
        connection.close()


def initialize(app):
    connection = connect(app.config["DATABASE_PATH"])
    try:
        for table, required in REQUIRED_COLUMNS.items():
            columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if columns and not required.issubset(columns):
                raise RuntimeError("Existing database has an incompatible schema; select a new ACOS database path.")
        connection.executescript(SCHEMA)
        if app.config["SEED_DEMO"]:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
                    for selected_domain, records in DEMO_RECORDS.items():
                        _import_records(connection, selected_domain, records, is_demo=True)
    finally:
        connection.close()


def feature_name(aspect):
    return " ".join(aspect.casefold().split()) if aspect else "(implicit)"


def _import_records(connection, selected_domain, records, is_demo=False):
    counts = {"products": 0, "reviews": 0, "quads": 0, "records": len(records), "skipped_records": 0}
    for record in records:
        existing = connection.execute("SELECT name, domain FROM products WHERE product_id = ?",
                                      (record["product_id"],)).fetchone()
        if existing and (existing["domain"] != selected_domain or existing["name"] != record["product_name"]):
            raise ApiError("An existing product_id has a different name or domain; use a distinct product_id.",
                           409, "product_conflict")
        if not existing:
            connection.execute("INSERT INTO products(product_id, name, domain, is_demo) VALUES(?, ?, ?, ?)",
                               (record["product_id"], record["product_name"], selected_domain, int(is_demo)))
            counts["products"] += 1
        cursor = connection.execute("INSERT OR IGNORE INTO reviews(product_id, text) VALUES(?, ?)",
                                    (record["product_id"], record["text"]))
        added_review = cursor.rowcount
        counts["reviews"] += added_review
        review_id = connection.execute("SELECT review_id FROM reviews WHERE product_id = ? AND text = ?",
                                       (record["product_id"], record["text"])).fetchone()[0]
        added_quads = 0
        for quad in record["quads"]:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO quads(review_id, aspect, category, opinion, sentiment, feature) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (review_id, quad["aspect"], quad["category"], quad["opinion"], quad["sentiment"],
                 feature_name(quad["aspect"])))
            added_quads += cursor.rowcount
        counts["quads"] += added_quads
        if not added_review and not added_quads:
            counts["skipped_records"] += 1
    return counts


def import_records(selected_domain, records):
    connection = get_db()
    with connection:
        connection.execute("BEGIN IMMEDIATE")
        return _import_records(connection, selected_domain, records)


SUMMARY_SELECT = """
SELECT p.product_id, p.name, p.domain, p.is_demo,
       COUNT(DISTINCT r.review_id) AS review_count, COUNT(q.quad_id) AS quad_count,
       COALESCE(AVG(CASE WHEN q.sentiment = 'Positive' THEN 1.0
                        WHEN q.quad_id IS NOT NULL THEN 0.0 END), 0.0) AS positive_ratio
FROM products p LEFT JOIN reviews r ON r.product_id = p.product_id
LEFT JOIN quads q ON q.review_id = r.review_id
"""


def product_dict(row):
    value = dict(row)
    value["is_demo"] = bool(value["is_demo"])
    value["positive_ratio"] = round(value["positive_ratio"], 4)
    return value


def products(selected_domain):
    rows = get_db().execute(SUMMARY_SELECT + " WHERE p.domain = ? GROUP BY p.product_id ORDER BY p.name, p.product_id",
                            (selected_domain,))
    return [product_dict(row) for row in rows]


def aggregate_features(selected_domain=None, selected_product=None):
    predicate, value = ("p.domain = ?", selected_domain) if selected_product is None else ("p.product_id = ?", selected_product)
    rows = get_db().execute(
        "SELECT q.category, q.feature, COUNT(*) AS count, "
        "SUM(q.sentiment = 'Positive') AS pos, SUM(q.sentiment = 'Neutral') AS neu, "
        "SUM(q.sentiment = 'Negative') AS neg "
        "FROM quads q JOIN reviews r ON q.review_id = r.review_id "
        "JOIN products p ON p.product_id = r.product_id WHERE " + predicate +
        " GROUP BY q.category, q.feature ORDER BY q.category, q.feature", (value,))
    result = []
    for row in rows:
        feature = dict(row)
        feature["score"] = round((feature["pos"] - feature["neg"]) / feature["count"], 4)
        result.append(feature)
    return result


def product_detail(selected_product):
    connection = get_db()
    product = connection.execute(SUMMARY_SELECT + " WHERE p.product_id = ? GROUP BY p.product_id",
                                 (selected_product,)).fetchone()
    if product is None:
        raise ApiError("Product was not found.", 404, "not_found")
    reviews = []
    for row in connection.execute("SELECT review_id, text FROM reviews WHERE product_id = ? ORDER BY review_id",
                                  (selected_product,)):
        review = dict(row)
        review["quads"] = [dict(quad) for quad in connection.execute(
            "SELECT aspect, category, opinion, sentiment FROM quads WHERE review_id = ? ORDER BY quad_id",
            (row["review_id"],))]
        reviews.append(review)
    return {"product": product_dict(product), "reviews": reviews,
            "features": aggregate_features(selected_product=selected_product)}


def recommendations(selected_domain, selections, mode, top_k):
    connection = get_db()
    clauses, parameters = [], [selected_domain]
    for category, feature in selections.items():
        clauses.append("(q.category = ? AND q.feature = ?)")
        parameters.extend((category, feature_name(feature)))
    rows = connection.execute(
        "SELECT p.product_id, r.review_id, r.text, q.category, q.feature, q.aspect, q.opinion, q.sentiment "
        "FROM quads q JOIN reviews r ON r.review_id = q.review_id "
        "JOIN products p ON p.product_id = r.product_id "
        "WHERE p.domain = ? AND (" + " OR ".join(clauses) + ") ORDER BY p.product_id, r.review_id, q.quad_id",
        parameters)
    matches = defaultdict(list)
    for row in rows:
        evidence = dict(row)
        matches[evidence.pop("product_id")].append(evidence)
    summaries = {product["product_id"]: product for product in products(selected_domain)}
    result = []
    for selected_product, evidence in matches.items():
        matched_categories = sorted({item["category"] for item in evidence})
        if mode == "AND" and len(matched_categories) != len(selections):
            continue
        score = sum({"Positive": 1, "Neutral": 0, "Negative": -1}[item["sentiment"]]
                    for item in evidence) / len(evidence)
        product = summaries[selected_product]
        result.append({**product, "score": round(score, 4), "matched_categories": matched_categories,
                       "evidence": evidence[:12], "evidence_count": len(evidence)})
    result.sort(key=lambda item: (-item["score"], -len(item["matched_categories"]),
                                  -item["review_count"], item["product_id"]))
    return result[:top_k]
