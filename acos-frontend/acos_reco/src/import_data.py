"""Parse an upload for preview; the backend remains the authority on import."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_RECORDS = 1000
SENTIMENTS = {"0": "Negative", "1": "Neutral", "2": "Positive",
              "negative": "Negative", "neutral": "Neutral", "positive": "Positive"}


class ImportFormatError(ValueError):
    pass


def _string(value: object, field: str, required: bool = False, limit: int = 200) -> str:
    if not isinstance(value, str):
        raise ImportFormatError(f"{field}: 문자열이 필요합니다.")
    value = value.strip()
    if required and not value:
        raise ImportFormatError(f"{field}: 빈 값을 사용할 수 없습니다.")
    if len(value) > limit:
        raise ImportFormatError(f"{field}: 최대 {limit}자까지 사용할 수 있습니다.")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise ImportFormatError(f"{field}: 지원하지 않는 제어 문자가 포함되어 있습니다.")
    return value


def _quad(value: object) -> dict:
    if not isinstance(value, dict) or set(value) - {"aspect", "category", "opinion", "sentiment"}:
        raise ImportFormatError("쿼드는 aspect, category, opinion, sentiment 필드를 가진 객체여야 합니다.")
    sentiment = SENTIMENTS.get(str(value.get("sentiment", "")).lower())
    if sentiment is None:
        raise ImportFormatError("sentiment는 Negative/Neutral/Positive 또는 0/1/2여야 합니다.")
    category = _string(value.get("category", ""), "category", True, 120)
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*(?:#[A-Z][A-Z0-9_]*)?", category):
        raise ImportFormatError("category: 대문자로 된 ACOS 카테고리 레이블이 필요합니다.")
    return {"aspect": _string(value.get("aspect", ""), "aspect"), "category": category,
            "opinion": _string(value.get("opinion", ""), "opinion", limit=500), "sentiment": sentiment}


def _record(value: object) -> dict:
    if not isinstance(value, dict) or set(value) - {"product_id", "product_name", "text", "quads"}:
        raise ImportFormatError("각 레코드는 product_id, product_name, text, quads 필드를 가진 객체여야 합니다.")
    quads = value.get("quads", [])
    if not isinstance(quads, list) or len(quads) > 64:
        raise ImportFormatError("quads는 최대 64개 객체를 포함하는 배열이어야 합니다.")
    product_id = _string(value.get("product_id", ""), "product_id", True, 80)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", product_id):
        raise ImportFormatError("product_id: 영문·숫자로 시작하는 영문, 숫자, 밑줄, 마침표, 하이픈만 사용할 수 있습니다.")
    return {"product_id": product_id,
            "product_name": _string(value.get("product_name", product_id), "product_name", True),
            "text": _string(value.get("text", ""), "text", True, 8000),
            "quads": [_quad(quad) for quad in quads]}


def _span_text(span: str, tokens: list[str]) -> str:
    if not re.fullmatch(r"-?\d+,-?\d+", span):
        raise ImportFormatError(f"잘못된 ACOS 토큰 범위: {span}")
    start, end = map(int, span.split(","))
    if (start, end) == (-1, -1):
        return ""
    if not 0 <= start < end <= len(tokens):
        raise ImportFormatError(f"ACOS 범위는 [시작, 끝) 규칙을 따라야 합니다: {span}")
    return " ".join(tokens[start:end])


def _acos_tsv(rows: list[list[str]], domain: str) -> tuple[list[dict], list[str]]:
    records = []
    generated = False
    for line_no, row in enumerate(rows, 1):
        if len(row) < 2:
            raise ImportFormatError(f"{line_no}행: 문장 뒤에 탭으로 구분된 ACOS 쿼드가 필요합니다.")
        text = row[0].strip()
        if " @@@ " in text:
            product_id, text = text.split(" @@@ ", 1)
        else:
            product_id = "import-" + domain + "-" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            generated = True
        tokens = text.split()
        quads = []
        for token in row[1:]:
            parts = token.split()
            if len(parts) != 4:
                raise ImportFormatError(f"{line_no}행: 'a_start,a_end CATEGORY SENTIMENT o_start,o_end' 형식이 필요합니다.")
            aspect, category, sentiment, opinion = parts
            try:
                quads.append({"aspect": _span_text(aspect, tokens), "category": category,
                              "sentiment": sentiment, "opinion": _span_text(opinion, tokens)})
            except ImportFormatError as exc:
                raise ImportFormatError(f"{line_no}행: {exc}") from exc
        records.append(_record({"product_id": product_id, "product_name": product_id,
                                "text": text, "quads": quads}))
    notices = ["ACOS 토큰 범위는 끝 인덱스를 포함하지 않습니다. (-1,-1)은 암시적 표현입니다."]
    if generated:
        notices.append("원본에 상품 ID가 없는 문장은 전체 문장의 해시로 각각 별도 상품에 연결됩니다. 실제 상품 단위 추천에는 상품 ID가 있는 JSON/TSV를 사용하세요.")
    return records, notices


def serialized_request_size(domain: str, records: list[dict]) -> int:
    """Match requests' JSON encoding, including ASCII escapes and separators."""
    return len(json.dumps({"domain": domain, "records": records},
                          ensure_ascii=True, allow_nan=False).encode("utf-8"))


def parse_upload(data: bytes, filename: str, domain: str) -> tuple[list[dict], list[str]]:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImportFormatError("업로드는 5 MiB 이하로 제한됩니다.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ImportFormatError("UTF-8로 저장한 JSON 또는 TSV 파일을 사용하세요.") from exc
    if not text.strip():
        raise ImportFormatError("파일이 비어 있습니다.")
    notices = []
    try:
        if filename.lower().endswith(".json"):
            values = json.loads(text)
            if isinstance(values, dict):
                if values.get("domain", domain) != domain:
                    raise ImportFormatError("파일의 domain이 현재 선택한 도메인과 다릅니다.")
                values = values.get("records")
            if not isinstance(values, list):
                raise ImportFormatError("JSON은 레코드 배열 또는 {\"records\": [...]} 객체여야 합니다.")
            records = [_record(value) for value in values]
        elif filename.lower().endswith(".tsv"):
            rows = [row for row in csv.reader(io.StringIO(text), delimiter="\t") if any(cell.strip() for cell in row)]
            if rows and {"product_id", "text", "quads"}.issubset(set(rows[0])):
                header = rows[0]
                if len(set(header)) != len(header) or set(header) - {"product_id", "product_name", "text", "quads"}:
                    raise ImportFormatError("TSV 헤더는 product_id, product_name(선택), text, quads를 사용하세요.")
                records = []
                for line_no, row in enumerate(rows[1:], 2):
                    if len(row) != len(header):
                        raise ImportFormatError(f"{line_no}행: 헤더와 열 개수가 다릅니다.")
                    value = dict(zip(header, row))
                    value["quads"] = json.loads(value["quads"])
                    records.append(_record(value))
            else:
                records, notices = _acos_tsv(rows, domain)
        else:
            raise ImportFormatError("JSON 또는 TSV 확장자를 사용하세요.")
    except (json.JSONDecodeError, csv.Error) as exc:
        raise ImportFormatError(f"파일 내용을 읽을 수 없습니다: {exc}") from exc
    if not 1 <= len(records) <= MAX_RECORDS:
        raise ImportFormatError("한 번에 1~1,000개의 리뷰 레코드를 가져올 수 있습니다.")
    return records, notices
