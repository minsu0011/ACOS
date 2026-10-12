"""Validate untrusted API inputs before database or predictor work."""
import re

DOMAINS = ("laptop", "rest16")
SENTIMENTS = ("Positive", "Neutral", "Negative")
MAX_RECORDS, MAX_QUADS, MAX_TEXT = 1000, 64, 8000


class ApiError(Exception):
    def __init__(self, message, status=400, code="invalid_request"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


def object_fields(value, allowed, required=(), label="request"):
    if not isinstance(value, dict):
        raise ApiError(f"{label} must be a JSON object.")
    if set(value) - set(allowed):
        raise ApiError(f"{label} contains unsupported fields.")
    if set(required) - set(value):
        raise ApiError(f"{label} is missing required fields.")
    return value


def text(value, label, limit=200, allow_empty=False):
    if not isinstance(value, str):
        raise ApiError(f"{label} must be a string.")
    value = value.strip()
    if len(value) > limit or (not value and not allow_empty):
        raise ApiError(f"{label} must contain {'0' if allow_empty else '1'} to {limit} characters.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ApiError(f"{label} contains unsupported control characters.")
    return value


def domain(value):
    if not isinstance(value, str) or value not in DOMAINS:
        raise ApiError("domain must be laptop or rest16.")
    return value


def product_id(value):
    value = text(value, "product_id", 80)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ApiError("product_id must use letters, digits, underscores, hyphens or periods.")
    return value


def category(value):
    value = text(value, "category", 120)
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*(?:#[A-Z][A-Z0-9_]*)?", value):
        raise ApiError("category must be an uppercase ACOS category label.")
    return value


def quads(value, label="quads"):
    if not isinstance(value, list) or len(value) > MAX_QUADS:
        raise ApiError(f"{label} must contain at most {MAX_QUADS} quadruples.")
    result = []
    for item in value:
        object_fields(item, ("aspect", "category", "opinion", "sentiment"),
                      ("aspect", "category", "opinion", "sentiment"), "quad")
        sentiment = item["sentiment"]
        if not isinstance(sentiment, str) or sentiment not in SENTIMENTS:
            raise ApiError("sentiment must be Positive, Neutral or Negative.")
        result.append({"aspect": text(item["aspect"], "aspect", allow_empty=True),
                       "category": category(item["category"]),
                       "opinion": text(item["opinion"], "opinion", 500, allow_empty=True),
                       "sentiment": sentiment})
    return result


def import_request(value):
    object_fields(value, ("domain", "records"), ("domain", "records"))
    selected_domain = domain(value["domain"])
    records = value["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
        raise ApiError(f"records must contain 1 to {MAX_RECORDS} review objects.")
    checked = []
    for record in records:
        object_fields(record, ("product_id", "product_name", "text", "quads"),
                      ("product_id", "product_name", "text", "quads"), "record")
        checked.append({"product_id": product_id(record["product_id"]),
                        "product_name": text(record["product_name"], "product_name"),
                        "text": text(record["text"], "text", MAX_TEXT),
                        "quads": quads(record["quads"])})
    return selected_domain, checked
