from __future__ import annotations
from collections import Counter
import hashlib
from typing import List, Dict, Tuple

from .types import ProductAgg


def sentence_to_product_id(text: str) -> str:
    """Create a stable review identity when a real product ID is unavailable."""
    if not isinstance(text, str):
        raise ValueError("Review text must be a string.")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"review_{digest}"


def build_product_index(rows: List[Dict]) -> List[ProductAgg]:
    # group by (product_id, category, feature)
    buckets: Dict[Tuple[str, str, str], List[int]] = {}
    for r in rows:
        product_id = r.get("product_id")
        pid = (
            str(product_id)
            if product_id is not None and str(product_id).strip()
            else sentence_to_product_id(r["text"])
        )
        sentiment = r["sentiment"]
        if type(sentiment) is not int or sentiment not in (0, 1, 2):
            raise ValueError("Sentiment must be the integer 0, 1, or 2.")
        key = (pid, str(r["category"]), str(r["feature"]))
        buckets.setdefault(key, []).append(sentiment)

    out: List[ProductAgg] = []
    for (pid, cat, feat), sentis in buckets.items():
        cnt = Counter(sentis)
        pos = int(cnt.get(2, 0))
        neu = int(cnt.get(1, 0))
        neg = int(cnt.get(0, 0))
        total = max(1, pos + neu + neg)
        score = (pos - neg) / total
        out.append(ProductAgg(
            product_id=str(pid),
            category=str(cat),
            feature=str(feat),
            pos=pos,
            neu=neu,
            neg=neg,
            score=float(score),
        ))
    return out
