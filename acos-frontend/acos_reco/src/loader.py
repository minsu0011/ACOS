from __future__ import annotations
import re
from typing import List, Tuple, Iterable, Dict

from .types import Quad

TSV_LINE_SPLIT = re.compile(r"\t+")
SPAN_RE = re.compile(r"^(?P<s>-?\d+),(?P<e>-?\d+)$")


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
    return a_span, category.strip(), sentiment, o_span


def parse_tsv_lines(lines: Iterable[str]) -> List[Quad]:
    quads: List[Quad] = []
    for raw in lines:
        raw = raw.strip("\n")
        if not raw:
            continue
        cols = TSV_LINE_SPLIT.split(raw)
        if not cols:
            continue
        text = cols[0]
        quad_tokens = cols[1:]
        for tok in quad_tokens:
            a_span, cat, senti, o_span = parse_quad_token(tok)
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
        a_s, a_e = q["aspect_span"]
        o_s, o_e = q["opinion_span"]
        def span_to_text(s: int, e: int) -> str:
            if s == -1 and e == -1:
                return ""
            toks = text.split()
            if 0 <= s < len(toks) and 0 <= e < len(toks):
                return " ".join(toks[s:e+1])
            return ""
        a_txt = span_to_text(a_s, a_e)
        o_txt = span_to_text(o_s, o_e)
        feat = (a_txt or o_txt or "implicit").lower()
        rows.append({
            "text": text,
            "category": q["category"],
            "sentiment": q["sentiment"],
            "feature": feat,
            "aspect_text": a_txt,
            "opinion_text": o_txt,
        })
    return rows
