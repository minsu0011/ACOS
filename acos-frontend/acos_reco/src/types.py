from typing import Dict, List, Tuple, Optional, TypedDict

Sentiment = int  # 0=neg, 1=neu, 2=pos

class Quad(TypedDict):
    text: str
    aspect_span: Tuple[int, int]  # (start_idx, end_idx) or (-1, -1) for implicit
    category: str                 # e.g., FOOD#QUALITY
    sentiment: Sentiment          # 0/1/2
    opinion_span: Tuple[int, int] # (start_idx, end_idx) or (-1, -1)

class ProductAgg(TypedDict):
    product_id: str
    category: str
    feature: str  # normalized aspect/opinion token
    pos: int
    neu: int
    neg: int
    score: float

DEFAULT_DOMAIN_CATS: Dict[str, List[str]] = {
    "rest16": [
        "RESTAURANT#GENERAL",
        "SERVICE#GENERAL",
        "FOOD#QUALITY",
        "FOOD#STYLE_OPTIONS",
        "FOOD#PRICES",
        "DRINKS#QUALITY",
        "DRINKS#PRICES",
        "DRINKS#STYLE_OPTIONS",
        "AMBIENCE#GENERAL",
        "LOCATION#GENERAL",
        "RESTAURANT#PRICES",
        "RESTAURANT#MISCELLANEOUS",
    ],
    "laptop": [
        "LAPTOP#GENERAL",
        "LAPTOP#OPERATION_PERFORMANCE",
        "LAPTOP#DESIGN_FEATURES",
        "LAPTOP#BATTERY",
        "LAPTOP#PORTABILITY",
        "LAPTOP#PRICE",
        "LAPTOP#QUALITY",
        "LAPTOP#USABILITY",
        "LAPTOP#CONNECTIVITY",
        "LAPTOP#DISPLAY",
    ],
}
