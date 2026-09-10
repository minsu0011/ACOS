from __future__ import annotations
from collections import Counter
from typing import List, Dict, Tuple

from .types import ProductAgg


def sentence_to_product_id(text: str) -> str:
    base = text[:40].lower()
    return f"prod_{abs(hash(base)) % 100000}"


def build_product_index(rows: List[Dict]) -> List[ProductAgg]:
    # group by (product_id, category, feature)
    buckets: Dict[Tuple[str, str, str], List[int]] = {}
    for r in rows:
        pid = sentence_to_product_id(r["text"])  # demo mapping
        key = (pid, str(r["category"]), str(r["feature"]))
        buckets.setdefault(key, []).append(int(r["sentiment"]))

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
