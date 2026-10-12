from __future__ import annotations
import re
from typing import List, Tuple, Iterable, Dict

from .types import Quad

TSV_LINE_SPLIT = re.compile(r"\t+")
SPAN_RE = re.compile(r"^(?P<s>-?\d+),(?P<e>-?\d+)$")


def _validate_span(span: Tuple[int, int], token_count: int | None = None) -> None:
    """ACOS spans are [start, end), with (-1, -1) for implicit terms."""
    if not isinstance(span, (tuple, list)) or len(span) != 2 or any(type(value) is not int for value in span):
        raise ValueError("A span must contain two integer indices.")
    start, end = span
    if (start, end) == (-1, -1):
        return
    if start < 0 or end <= start:
        raise ValueError(f"Invalid end-exclusive span: {span}")
    if token_count is not None and end > token_count:
        raise ValueError(f"Span {span} exceeds the {token_count} review tokens.")


def _validate_sentiment(sentiment: int) -> None:
    if type(sentiment) is not int or sentiment not in (0, 1, 2):
        raise ValueError("Sentiment must be 0 (negative), 1 (neutral), or 2 (positive).")


def parse_quad_token(tok: str) -> Tuple[Tuple[int, int], str, int, Tuple[int, int]]:
    # token pattern: "a_start,a_end CATEGORY SENTIMENT o_start,o_end"
    parts = tok.split()
    if len(parts) != 4:
        raise ValueError(f"Bad quad token: {tok}")
    a_span_s, category, senti_s, o_span_s = parts
    a_m = SPAN_RE.match(a_span_s)
    o_m = SPAN_RE.match(o_span_s)
    if not a_m or not o_m:
        raise ValueError(f"Bad span in token: {tok}")
    a_span = (int(a_m.group("s")), int(a_m.group("e")))
    o_span = (int(o_m.group("s")), int(o_m.group("e")))
    sentiment = int(senti_s)
    _validate_span(a_span)
    _validate_span(o_span)
    _validate_sentiment(sentiment)
    return a_span, category.strip(), sentiment, o_span


def parse_tsv_lines(lines: Iterable[str]) -> List[Quad]:
    quads: List[Quad] = []
    for raw in lines:
        raw = raw.rstrip("\r\n")
        if not raw:
            continue
        cols = TSV_LINE_SPLIT.split(raw)
        if not cols:
            continue
        text = cols[0]
        quad_tokens = cols[1:]
        token_count = len(text.split())
        for tok in quad_tokens:
            a_span, cat, senti, o_span = parse_quad_token(tok)
            _validate_span(a_span, token_count)
            _validate_span(o_span, token_count)
            quads.append(Quad(
                text=text,
                aspect_span=a_span,
                category=cat,
                sentiment=senti,
                opinion_span=o_span,
            ))
    return quads


def load_acos_tsv(path: str) -> List[Quad]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return parse_tsv_lines(lines)


def to_rows(quads: List[Quad]) -> List[Dict]:
    rows: List[Dict] = []
    for q in quads:
        text = q["text"]
        if not isinstance(text, str):
            raise ValueError("Review text must be a string.")
        toks = text.split()
        _validate_span(q["aspect_span"], len(toks))
        _validate_span(q["opinion_span"], len(toks))
        _validate_sentiment(q["sentiment"])
        a_s, a_e = q["aspect_span"]
        o_s, o_e = q["opinion_span"]
        def span_to_text(s: int, e: int) -> str:
            if s == -1 and e == -1:
                return ""
            return " ".join(toks[s:e])
        a_txt = span_to_text(a_s, a_e)
        o_txt = span_to_text(o_s, o_e)
        feat = (a_txt or o_txt or "implicit").lower()
        row = {
            "text": text,
            "category": q["category"],
            "sentiment": q["sentiment"],
            "feature": feat,
            "aspect_text": a_txt,
            "opinion_text": o_txt,
        }
        if q.get("product_id") is not None:
            row["product_id"] = q["product_id"]
        rows.append(row)
    return rows
