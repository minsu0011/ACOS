"""Frontend contracts and state transitions, without a running HTTP server."""
import csv
import io
import json
import sys
from pathlib import Path

import pytest
import requests

FRONTEND = Path(__file__).resolve().parents[1] / "acos-frontend" / "acos_reco"
sys.path.insert(0, str(FRONTEND))

from src.api_client import ApiClient, ApiError
from src.import_data import ImportFormatError, MAX_UPLOAD_BYTES, parse_upload, serialized_request_size


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse({"status": "ok"})
        self.error = error
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.response


def sample_record(**changes):
    value = {"product_id": "my-laptop", "product_name": "내 상품",
             "text": "The battery is great.", "quads": [
                 {"aspect": "battery", "category": "BATTERY#OPERATION_PERFORMANCE",
                  "opinion": "great", "sentiment": 2}]}
    value.update(changes)
    return value


def encoded_json(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def test_client_environment_timeout_and_encoded_product_path(monkeypatch):
    monkeypatch.setenv("ACOS_API_URL", "http://example.test/prefix/")
    monkeypatch.setenv("ACOS_API_TIMEOUT", "7")
    session = FakeSession()
    client = ApiClient(session=session)
    client.product("a/b with spaces")
    args, kwargs = session.calls[0]
    assert args == ("GET", "http://example.test/prefix/api/products/a%2Fb%20with%20spaces")
    assert kwargs["timeout"] == 7
    assert kwargs["headers"]["Accept"] == "application/json"


@pytest.mark.parametrize("address", ["file:///tmp/a", "https://user:secret@host.test",
                                     "http://host.test?q=x", "http://host.test#fragment", "not-a-url", ""])
def test_client_rejects_invalid_addresses(address):
    with pytest.raises(ValueError):
        ApiClient(address)


@pytest.mark.parametrize("timeout", [0, 61, "bad", float("nan"), float("inf")])
def test_client_rejects_invalid_timeouts(timeout):
    with pytest.raises(ValueError):
        ApiClient(timeout=timeout)


@pytest.mark.parametrize("exception, wording", [(requests.Timeout(), "시간"),
                                               (requests.ConnectionError(), "연결")])
def test_client_errors_are_actionable_and_post_is_not_retried(exception, wording):
    session = FakeSession(error=exception)
    with pytest.raises(ApiError, match=wording):
        ApiClient(session=session).import_records("laptop", [sample_record()])
    assert len(session.calls) == 1


def test_client_reports_validation_and_non_json_errors():
    with pytest.raises(ApiError, match="invalid category") as error:
        ApiClient(session=FakeSession(FakeResponse({"error": "invalid category"}, 400))).analyze("review", "laptop")
    assert error.value.status_code == 400
    with pytest.raises(ApiError, match="JSON"):
        ApiClient(session=FakeSession(FakeResponse(ValueError("not JSON")))).health()
    with pytest.raises(ApiError, match="객체"):
        ApiClient(session=FakeSession(FakeResponse([]))).meta()


def test_client_plain_text_413_reports_request_size_without_json_decode():
    response = FakeResponse(AssertionError("An HTTP 413 must be handled before JSON decoding."), 413)
    session = FakeSession(response)
    with pytest.raises(ApiError, match="요청 크기") as error:
        ApiClient(session=session).import_records("laptop", [sample_record()])
    assert error.value.status_code == 413
    assert len(session.calls) == 1


def test_client_posts_domain_and_selection_contract():
    session = FakeSession()
    client = ApiClient("http://localhost:5001", 10, session)
    client.analyze("The food is good.", "rest16")
    client.recommend("rest16", {"FOOD#QUALITY": "food"}, "AND", 3)
    client.import_records("rest16", [])
    assert session.calls[0][1]["json"] == {"sentence": "The food is good.", "domain": "rest16"}
    assert session.calls[1][1]["json"] == {"domain": "rest16", "selections": {"FOOD#QUALITY": "food"},
                                          "mode": "AND", "top_k": 3}
    assert session.calls[2][0][1].endswith("/api/import")


def test_json_upload_bom_normalizes_sentiment_and_preserves_product():
    records, notices = parse_upload(b"\xef\xbb\xbf" + encoded_json({"domain": "laptop", "records": [sample_record()]}),
                                    "reviews.JSON", "laptop")
    assert records[0]["product_id"] == "my-laptop"
    assert records[0]["quads"][0]["sentiment"] == "Positive"
    assert not notices
    assert parse_upload(encoded_json([sample_record(quads=[])]), "a.json", "laptop")[0][0]["quads"] == []


def test_upload_rejects_domain_mix_invalid_fields_and_limits():
    with pytest.raises(ImportFormatError, match="domain"):
        parse_upload(encoded_json({"domain": "rest16", "records": [sample_record()]}), "a.json", "laptop")
    for record in [sample_record(product_id="../escape"), sample_record(text="x" * 8001),
                   sample_record(product_name="x" * 201), sample_record(quads=[{}]),
                   sample_record(extra="unsupported")]:
        with pytest.raises(ImportFormatError):
            parse_upload(encoded_json([record]), "a.json", "laptop")
    with pytest.raises(ImportFormatError, match="1,000"):
        parse_upload(encoded_json([sample_record()] * 1001), "a.json", "laptop")
    with pytest.raises(ImportFormatError, match="5 MiB"):
        parse_upload(b"x" * (MAX_UPLOAD_BYTES + 1), "a.json", "laptop")


def test_header_tsv_supports_json_quoted_quads():
    output = io.StringIO()
    writer = csv.writer(output, delimiter="\t")
    record = sample_record()
    writer.writerow(["product_id", "product_name", "text", "quads"])
    writer.writerow([record["product_id"], record["product_name"], record["text"], json.dumps(record["quads"])])
    records, notices = parse_upload(output.getvalue().encode("utf-8"), "reviews.tsv", "laptop")
    assert records[0]["quads"][0]["sentiment"] == "Positive"
    assert records[0]["product_name"] == "내 상품"
    assert not notices


def test_official_tsv_end_exclusive_last_token_implicit_stable_ids():
    raw = b"The battery is great.\t1,2 BATTERY#OPERATION_PERFORMANCE 2 3,4\t-1,-1 LAPTOP#GENERAL 1 -1,-1\n"
    first, notices = parse_upload(raw, "a.tsv", "laptop")
    second, _ = parse_upload(raw, "b.tsv", "laptop")
    assert first == second
    assert first[0]["quads"][0]["aspect"] == "battery"
    assert first[0]["quads"][0]["opinion"] == "great."
    assert first[0]["quads"][1]["aspect"] == first[0]["quads"][1]["opinion"] == ""
    assert "상품 ID" in notices[1]
    prefixed, _ = parse_upload(b"known-id @@@ The battery is great.\t1,2 BATTERY#QUALITY 2 3,4", "a.tsv", "laptop")
    assert prefixed[0]["product_id"] == "known-id"
    different, _ = parse_upload(raw.replace(b"great.", b"poor."), "a.tsv", "laptop")
    assert first[0]["product_id"] != different[0]["product_id"]


@pytest.mark.parametrize("span", ["1,1", "-1,2", "3,5", "2,1", "1,x"])
def test_official_tsv_invalid_span_reports_line(span):
    with pytest.raises(ImportFormatError, match="1행"):
        parse_upload(f"The battery is good.\t{span} BATTERY#QUALITY 2 3,4".encode(), "a.tsv", "laptop")


def test_request_size_matches_requests_ascii_serialization():
    records, _ = parse_upload(encoded_json([sample_record()]), "a.json", "laptop")
    prepared = requests.Request("POST", "http://localhost/api/import",
                                json={"domain": "laptop", "records": records}).prepare()
    assert serialized_request_size("laptop", records) == len(prepared.body)
    assert serialized_request_size("laptop", records) > len(encoded_json({"domain": "laptop", "records": records}))


class FakeApiClient:
    calls = []
    mode = "heuristic"
    loaded = False
    unavailable = False

    def __init__(self, *args, **kwargs):
        pass

    def health(self):
        if self.unavailable:
            raise ApiError("백엔드에 연결할 수 없습니다.")
        return {"status": "ok", "analysis_mode": self.mode, "model_loaded": self.loaded}

    def meta(self):
        return {"domains": ["laptop", "rest16"], "analysis_mode": self.mode,
                "limits": {"body_bytes": 2_000_000}}

    def products(self, domain):
        return {"products": [{"product_id": "p-" + domain, "name": "[상품](https://example.test)",
                              "domain": domain, "review_count": 1, "quad_count": 1,
                              "positive_ratio": 1.0, "is_demo": True}]}

    def product(self, product_id):
        domain = product_id[2:]
        return {"product": self.products(domain)["products"][0],
                "reviews": [{"review_id": "r1", "text": "<script>alert('review')</script>",
                             "quads": [{"aspect": "battery", "category": "BATTERY#QUALITY",
                                        "opinion": "great", "sentiment": "Positive"}]}], "features": []}

    def categories(self, domain):
        category, feature = ("BATTERY#QUALITY", "battery") if domain == "laptop" else ("FOOD#QUALITY", "food")
        return {"categories": [{"category": category, "features": [{"feature": feature, "count": 2}]}]}

    def recommend(self, domain, selections, mode, top_k):
        self.calls.append(("recommend", domain, selections, mode, top_k))
        return {"items": [{**self.products(domain)["products"][0], "score": 1.0,
                           "matched_categories": list(selections),
                           "evidence": [{"category": next(iter(selections)), "feature": next(iter(selections.values())),
                                         "aspect": "battery", "opinion": "great", "sentiment": "Positive",
                                         "review_id": "r1", "text": "The battery is great."}]}]}

    def analyze(self, sentence, domain):
        self.calls.append(("analyze", sentence, domain))
        return {"input_sentence": sentence, "analysis_mode": self.mode,
                "results": [{"aspect": "battery", "category": "BATTERY#QUALITY",
                             "opinion": "great", "sentiment": "Positive"}]}

    def import_records(self, domain, records):
        self.calls.append(("import", domain, records))
        return {"imported": {"records": len(records), "reviews": len(records)}}


@pytest.fixture
def app_test(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    import src.api_client as module
    monkeypatch.setattr(module, "ApiClient", FakeApiClient)
    FakeApiClient.calls = []
    FakeApiClient.mode = "heuristic"
    FakeApiClient.loaded = False
    FakeApiClient.unavailable = False
    return testing.AppTest.from_file(str(FRONTEND / "streamlit_app.py"), default_timeout=15)


def button_with_label(app, label):
    return next(button for button in app.button if button.label == label)


def test_ui_backend_recommendation_detail_and_domain_reset(app_test):
    app = app_test.run()
    assert not app.exception
    assert app.metric[0].value == "1"
    assert any("\\[상품\\]" in heading.value for heading in app.subheader)
    app.radio(key="app_view").set_value("조건 추천").run()
    app.selectbox(key="selection_BATTERY#QUALITY").set_value("battery")
    app.radio(key="recommendation_mode").set_value("AND")
    app.slider(key="recommendation_top_k").set_value(4)
    button_with_label(app, "추천 결과 보기").click().run()
    assert not app.exception
    assert FakeApiClient.calls[-1] == ("recommend", "laptop", {"BATTERY#QUALITY": "battery"}, "AND", 4)
    assert any(metric.value == "+1.00" for metric in app.metric)
    app.button(key="recommendation_open_p-laptop").click().run()
    assert app.session_state["app_view"] == "상품 탐색"
    assert any("<script>alert" in element.value for element in app.text)
    assert not any("<script>alert" in element.value for element in app.markdown)
    app.radio(key="domain").set_value("rest16").run()
    assert not app.exception
    assert "detail_product" not in app.session_state
    assert "recommendation_result" not in app.session_state
    assert "recommendation_mode" not in app.session_state
    assert "recommendation_top_k" not in app.session_state
    assert "p-rest16" in [caption.value for caption in app.caption]


def test_ui_analysis_tracks_domain_and_clears_result(app_test):
    app = app_test.run()
    app.radio(key="app_view").set_value("리뷰 분석").run()
    assert any("학습된 ACOS 모델 추론 결과가 아니" in warning.value for warning in app.warning)
    app.text_area(key="analysis_sentence").set_value("The battery is great.")
    button_with_label(app, "리뷰 분석하기").click().run()
    assert not app.exception
    assert FakeApiClient.calls[-1] == ("analyze", "The battery is great.", "laptop")
    assert len(app.dataframe) == 1
    app.radio(key="domain").set_value("rest16").run()
    assert "analysis_result" not in app.session_state
    assert "food" in app.text_area(key="analysis_sentence").value


def test_ui_connection_error_has_retry_without_crash(app_test):
    FakeApiClient.unavailable = True
    app = app_test.run()
    assert not app.exception
    assert any("연결" in error.value for error in app.error)
    assert button_with_label(app, "다시 연결")
    FakeApiClient.unavailable = False
    button_with_label(app, "다시 연결").click().run()
    assert not app.exception
    assert len(app.metric) > 0


def test_ui_missing_model_is_not_claimed_as_heuristic(app_test):
    FakeApiClient.mode = "model"
    app = app_test.run()
    app.radio(key="app_view").set_value("리뷰 분석").run()
    assert not app.exception
    assert any("모델을 불러오지 못" in error.value for error in app.error)
    assert not any("규칙 기준선" in warning.value for warning in app.warning)
    assert button_with_label(app, "리뷰 분석하기").disabled


def test_ui_upload_view_has_explicit_import_preview_guidance(app_test):
    app = app_test.run()
    app.radio(key="app_view").set_value("데이터 가져오기").run()
    assert not app.exception
    assert any("미리보기" in caption.value for caption in app.caption)
    assert not any(button.label == "백엔드에 가져오기" for button in app.button)


class UploadedFixture:
    name = "reviews.json"

    def __init__(self, data):
        self.data = data

    def getvalue(self):
        return self.data


def test_ui_upload_preview_requires_explicit_button(app_test, monkeypatch):
    import streamlit
    upload = UploadedFixture(encoded_json([sample_record()]))
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: upload)
    app = app_test.run()
    app.radio(key="app_view").set_value("데이터 가져오기").run()
    assert not app.exception
    assert not FakeApiClient.calls
    assert len(app.dataframe) == 1
    app.button(key="import_submit").click().run()
    assert not app.exception
    assert FakeApiClient.calls[0][0:2] == ("import", "laptop")
    assert FakeApiClient.calls[0][2][0]["quads"][0]["sentiment"] == "Positive"
    assert any("완료" in message.value for message in app.success)


def test_ui_oversized_serialized_import_is_disabled_without_request(app_test, monkeypatch):
    import streamlit
    source = encoded_json([sample_record(text="한" * 2000)] * 200)
    assert len(source) < MAX_UPLOAD_BYTES
    upload = UploadedFixture(source)
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: upload)
    app = app_test.run()
    app.radio(key="app_view").set_value("데이터 가져오기").run()
    assert not app.exception
    assert app.button(key="import_submit").disabled
    assert any("용량 제한" in error.value for error in app.error)
    assert not FakeApiClient.calls


def test_ui_invalid_upload_error_escapes_markdown(app_test, monkeypatch):
    import streamlit
    upload = UploadedFixture(b"The battery is great.\t[click](https://example.test) BATTERY#QUALITY 2 3,4")
    upload.name = "invalid.tsv"
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: upload)
    app = app_test.run()
    app.radio(key="app_view").set_value("데이터 가져오기").run()
    assert not app.exception
    assert any("\\[click\\]" in error.value for error in app.error)
    assert not FakeApiClient.calls


def test_ui_manual_navigation_from_detail_returns_to_catalog(app_test):
    app = app_test.run()
    app.button(key="catalog_open_p-laptop").click().run()
    assert app.session_state["detail_product"] == "p-laptop"
    app.radio(key="app_view").set_value("리뷰 분석").run()
    assert "detail_product" not in app.session_state
    app.radio(key="app_view").set_value("상품 탐색").run()
    assert not app.exception
    assert app.metric[0].label == "등록 상품"
    assert not any("<script>alert" in element.value for element in app.text)


def test_ui_import_clears_detail_and_catalog_shows_list(app_test, monkeypatch):
    import streamlit
    upload = UploadedFixture(encoded_json([sample_record()]))
    monkeypatch.setattr(streamlit, "file_uploader", lambda *args, **kwargs: upload)
    app = app_test.run()
    app.button(key="catalog_open_p-laptop").click().run()
    app.radio(key="app_view").set_value("데이터 가져오기").run()
    assert "detail_product" not in app.session_state
    # A stale detail value must also be cleared by the successful import itself.
    app.session_state["detail_product"] = "p-laptop"
    app.button(key="import_submit").click().run()
    assert "detail_product" not in app.session_state
    assert FakeApiClient.calls[-1][0] == "import"
    app.radio(key="app_view").set_value("상품 탐색").run()
    assert not app.exception
    assert app.metric[0].label == "등록 상품"
