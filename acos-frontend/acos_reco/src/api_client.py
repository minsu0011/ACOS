"""Small, testable HTTP client for the shared ACOS backend.

POST requests are deliberately not retried: an import must never be repeated
without the user's explicit action.
"""
from __future__ import annotations

import math
import os
from typing import Any
from urllib.parse import quote, urlsplit

import requests


class ApiError(RuntimeError):
    """An actionable backend failure suitable for display in the UI."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None,
                 session: Any = None):
        self.base_url = (base_url if base_url is not None else os.environ.get("ACOS_API_URL", "http://127.0.0.1:5001")).strip().rstrip("/")
        parsed = urlsplit(self.base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("API 주소는 사용자 정보·쿼리가 없는 http 또는 https URL이어야 합니다.")
        try:
            self.timeout = float(timeout if timeout is not None else os.environ.get("ACOS_API_TIMEOUT", "10"))
        except (TypeError, ValueError) as exc:
            raise ValueError("API 제한 시간은 1~60초의 숫자로 입력하세요.") from exc
        if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 60:
            raise ValueError("API 제한 시간은 1~60초의 숫자로 입력하세요.")
        self.session = session if session is not None else requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            response = self.session.request(method, self.base_url + path,
                                            timeout=self.timeout,
                                            headers={"Accept": "application/json"}, **kwargs)
        except requests.Timeout as exc:
            raise ApiError("백엔드 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요.") from exc
        except requests.ConnectionError as exc:
            raise ApiError("백엔드에 연결할 수 없습니다. API 주소와 서버 실행 상태를 확인한 뒤 다시 연결하세요.") from exc
        except requests.RequestException as exc:
            raise ApiError("API 요청을 완료하지 못했습니다. 연결 설정을 확인하고 다시 시도하세요.") from exc
        # Waitress may reject an oversized body before Flask can return JSON.
        if response.status_code == 413:
            raise ApiError("API 요청 크기가 서버의 허용 용량을 초과했습니다. 파일을 나누거나 레코드를 줄여 다시 시도하세요.", 413)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiError("백엔드가 JSON 응답을 반환하지 않았습니다. API 주소를 확인하세요.", response.status_code) from exc
        if response.status_code >= 400:
            detail = payload.get("message") or payload.get("error") if isinstance(payload, dict) else None
            message = f"API 요청이 거절되었습니다 (HTTP {response.status_code})."
            if isinstance(detail, str):
                message += " " + detail[:500]
            raise ApiError(message, response.status_code)
        if not isinstance(payload, dict):
            raise ApiError("백엔드 응답 형식이 올바르지 않습니다. JSON 객체가 필요합니다.")
        return payload

    def health(self) -> dict:
        return self._request("GET", "/api/health")

    def meta(self) -> dict:
        return self._request("GET", "/api/meta")

    def products(self, domain: str) -> dict:
        return self._request("GET", "/api/products", params={"domain": domain})

    def product(self, product_id: str) -> dict:
        return self._request("GET", "/api/products/" + quote(product_id, safe=""))

    def categories(self, domain: str) -> dict:
        return self._request("GET", "/api/categories", params={"domain": domain})

    def analyze(self, sentence: str, domain: str) -> dict:
        return self._request("POST", "/api/analyze", json={"sentence": sentence, "domain": domain})

    def recommend(self, domain: str, selections: dict[str, str], mode: str, top_k: int) -> dict:
        return self._request("POST", "/api/recommend", json={"domain": domain,
                             "selections": selections, "mode": mode, "top_k": top_k})

    def import_records(self, domain: str, records: list[dict]) -> dict:
        return self._request("POST", "/api/import", json={"domain": domain, "records": records})
