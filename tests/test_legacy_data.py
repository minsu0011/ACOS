"""Compatibility checks for the original ACOS TSV and aggregation utilities."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "acos-frontend" / "acos_reco"
sys.path.insert(0, str(FRONTEND_ROOT))

from src.indexer import build_product_index, sentence_to_product_id
from src.loader import parse_quad_token, parse_tsv_lines, to_rows


def test_upstream_end_exclusive_spans_and_final_token():
    quads = parse_tsv_lines([
        "battery life is amazing\t0,2 BATTERY#QUALITY 2 3,4\r\n",
        "great screen\t1,2 DISPLAY#QUALITY 2 0,1\n",
    ])
    rows = to_rows(quads)
    assert rows[0]["aspect_text"] == "battery life"
    assert rows[0]["opinion_text"] == "amazing"
    assert rows[1]["aspect_text"] == "screen"
    assert rows[1]["opinion_text"] == "great"


def test_implicit_spans_remain_empty():
    rows = to_rows(parse_tsv_lines(["fine\t-1,-1 LAPTOP#GENERAL 1 -1,-1"]))
    assert rows[0]["aspect_text"] == rows[0]["opinion_text"] == ""
    assert rows[0]["feature"] == "implicit"


@pytest.mark.parametrize("span", ["-1,0", "0,-1", "-2,-2", "0,0", "2,1"])
def test_invalid_spans_are_rejected(span):
    with pytest.raises(ValueError):
        parse_quad_token(f"{span} LAPTOP#GENERAL 2 -1,-1")


@pytest.mark.parametrize("sentiment", ["-1", "3", "positive", "2.0"])
def test_invalid_sentiment_is_rejected(sentiment):
    with pytest.raises(ValueError):
        parse_quad_token(f"0,1 LAPTOP#GENERAL {sentiment} -1,-1")


def test_out_of_bounds_span_is_rejected():
    with pytest.raises(ValueError):
        parse_tsv_lines(["great screen\t1,3 DISPLAY#QUALITY 2 0,1"])


def test_direct_quads_are_also_validated():
    quad = {
        "text": "great screen",
        "aspect_span": (1, 2),
        "category": "DISPLAY#QUALITY",
        "sentiment": 3,
        "opinion_span": (0, 1),
    }
    with pytest.raises(ValueError):
        to_rows([quad])
    quad["sentiment"] = 2
    quad["aspect_span"] = (1, 3)
    with pytest.raises(ValueError):
        to_rows([quad])


def test_review_ids_use_full_text_and_are_stable_across_processes():
    text = "same prefix " * 4 + "first review"
    other = "same prefix " * 4 + "different review"
    expected = "review_" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert sentence_to_product_id(text) == expected
    assert sentence_to_product_id(other) != expected
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(FRONTEND_ROOT)}
        result = subprocess.run(
            [sys.executable, "-c", "from src.indexer import sentence_to_product_id; "
             "import sys; print(sentence_to_product_id(sys.argv[1]))", text],
            env=env, check=True, capture_output=True, text=True,
        )
        assert result.stdout.strip() == expected


def test_explicit_product_ids_survive_conversion_and_group_distinct_reviews():
    quads = parse_tsv_lines([
        "great screen\t1,2 DISPLAY#QUALITY 2 0,1",
        "bad screen\t1,2 DISPLAY#QUALITY 0 0,1",
    ])
    for quad in quads:
        quad["product_id"] = "SKU-123"
    rows = to_rows(quads)
    assert [row["product_id"] for row in rows] == ["SKU-123", "SKU-123"]
    aggregate = build_product_index(rows)
    assert len(aggregate) == 1
    assert aggregate[0]["product_id"] == "SKU-123"
    assert (aggregate[0]["pos"], aggregate[0]["neu"], aggregate[0]["neg"]) == (1, 0, 1)
    assert aggregate[0]["score"] == 0.0


def test_blank_product_id_falls_back_to_review_identity():
    rows = to_rows(parse_tsv_lines(["great screen\t1,2 DISPLAY#QUALITY 2 0,1"]))
    rows[0]["product_id"] = ""
    aggregate = build_product_index(rows)
    assert aggregate[0]["product_id"] == sentence_to_product_id("great screen")


@pytest.mark.parametrize("sentiment", [-1, 3, True, 2.5, "2"])
def test_aggregation_rejects_invalid_sentiment(sentiment):
    with pytest.raises(ValueError):
        build_product_index([{
            "text": "great screen", "category": "DISPLAY#QUALITY", "feature": "screen",
            "sentiment": sentiment,
        }])
