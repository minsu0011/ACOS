"""ACOS review intelligence UI. Analysis and ranking belong to the API."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st

from src.api_client import ApiClient, ApiError
from src.import_data import ImportFormatError, parse_upload, serialized_request_size

APP_ROOT = Path(__file__).resolve().parent
DOMAIN_LABELS = {"laptop": "노트북", "rest16": "레스토랑"}
SENTIMENT_LABELS = {"Positive": "긍정", "Neutral": "중립", "Negative": "부정"}
CAT_LABELS = {
    "LAPTOP#GENERAL": "노트북 · 종합", "LAPTOP#PRICE": "노트북 · 가격",
    "LAPTOP#QUALITY": "노트북 · 품질", "LAPTOP#PORTABILITY": "휴대성",
    "LAPTOP#DESIGN_FEATURES": "디자인", "LAPTOP#OPERATION_PERFORMANCE": "전체 성능",
    "LAPTOP#USABILITY": "사용성", "LAPTOP#CONNECTIVITY": "연결성",
    "LAPTOP#BATTERY": "배터리", "LAPTOP#DISPLAY": "디스플레이",
    "BATTERY#OPERATION_PERFORMANCE": "배터리 · 성능", "BATTERY#QUALITY": "배터리 · 품질",
    "CPU#OPERATION_PERFORMANCE": "CPU · 성능", "CPU#QUALITY": "CPU · 품질",
    "DISPLAY#QUALITY": "디스플레이 · 품질", "DISPLAY#USABILITY": "디스플레이 · 사용성",
    "GRAPHICS#OPERATION_PERFORMANCE": "그래픽 · 성능", "KEYBOARD#USABILITY": "키보드 · 사용성",
    "PORTS#USABILITY": "포트 · 사용성", "MEMORY#OPERATION_PERFORMANCE": "메모리 · 성능",
    "HARD_DISC#OPERATION_PERFORMANCE": "저장장치 · 성능", "HARDWARE#QUALITY": "하드웨어 · 품질",
    "FOOD#QUALITY": "음식 · 품질", "FOOD#PRICES": "음식 · 가격",
    "FOOD#STYLE_OPTIONS": "음식 · 메뉴 구성", "DRINKS#QUALITY": "음료 · 품질",
    "DRINKS#PRICES": "음료 · 가격", "DRINKS#STYLE_OPTIONS": "음료 · 메뉴 구성",
    "SERVICE#GENERAL": "서비스", "AMBIENCE#GENERAL": "분위기", "LOCATION#GENERAL": "위치",
    "RESTAURANT#GENERAL": "레스토랑 · 종합", "RESTAURANT#PRICES": "레스토랑 · 가격",
    "RESTAURANT#MISCELLANEOUS": "레스토랑 · 기타",
}
VIEWS = ["상품 탐색", "조건 추천", "리뷰 분석", "데이터 가져오기"]


def category_label(value: str) -> str:
    return CAT_LABELS.get(value, value.replace("#", " · "))


def literal_markdown(value: str) -> str:
    """Render imported text literally inside headings and captions."""
    return re.sub(r"([\\`*_{}\[\]()#+.!|<>~$\-])", r"\\\1", str(value))


def reset_context() -> None:
    """Never carry a product, upload, or recommendation into another domain."""
    prefixes = ("selection_", "analysis_", "recommendation_", "detail_", "import_", "catalog_")
    for key in list(st.session_state):
        if key.startswith(prefixes):
            del st.session_state[key]


def open_product(product_id: str) -> None:
    st.session_state["detail_product"] = product_id
    st.session_state["app_view"] = "상품 탐색"


def close_product() -> None:
    st.session_state.pop("detail_product", None)


def report_api_error(exc: ApiError) -> None:
    st.error(literal_markdown(str(exc)))
    st.caption("서버 실행 후 같은 버튼을 다시 누르거나, 사이드바의 ‘다시 연결’을 사용하세요.")


def json_download(label: str, payload: dict | list, filename: str, key: str) -> None:
    st.download_button(label, data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=filename, mime="application/json", key=key)


def quad_rows(quads: list[dict]) -> list[dict]:
    return [{"대상 (Aspect)": q.get("aspect") or "암시적",
             "카테고리 (Category)": q.get("category", ""),
             "의견 (Opinion)": q.get("opinion") or "암시적",
             "감성 (Sentiment)": SENTIMENT_LABELS.get(q.get("sentiment"), q.get("sentiment", ""))}
            for q in quads]


def sidebar() -> tuple[str, ApiClient | None]:
    with st.sidebar:
        st.markdown("### ACOS Studio")
        st.caption("리뷰에서 선택의 근거를 찾다")
        st.divider()
        domain = st.radio("탐색 도메인", list(DOMAIN_LABELS), format_func=DOMAIN_LABELS.get,
                          key="domain", on_change=reset_context)
        st.divider()
        with st.expander("API 연결 설정"):
            with st.form("api_settings"):
                address = st.text_input("백엔드 주소", value=st.session_state.get("api_url", os.environ.get("ACOS_API_URL", "http://127.0.0.1:5001")))
                raw_timeout = st.session_state.get("api_timeout", os.environ.get("ACOS_API_TIMEOUT", "10"))
                try:
                    default_timeout = min(60, max(1, int(float(raw_timeout))))
                except (TypeError, ValueError, OverflowError):
                    default_timeout = 10
                timeout = st.number_input("응답 제한 시간 (초)", min_value=1, max_value=60, value=default_timeout)
                if st.form_submit_button("연결 설정 적용"):
                    try:
                        ApiClient(address, timeout)
                    except ValueError as exc:
                        st.error(str(exc))
                    else:
                        st.session_state["api_url"] = address
                        st.session_state["api_timeout"] = timeout
                        reset_context()
            st.caption("환경변수 ACOS_API_URL / ACOS_API_TIMEOUT도 지원합니다.")
        st.button("다시 연결", width="stretch")
        st.divider()
        st.caption("2025 졸업 프로젝트")
        st.caption("Aspect · Category · Opinion · Sentiment")
        st.caption("예시 데이터와 규칙 기반 분석을 포함한 실행 가능한 데모입니다.")
    try:
        client = ApiClient(st.session_state.get("api_url"), st.session_state.get("api_timeout"))
    except ValueError as exc:
        st.error(str(exc))
        st.info("사이드바에서 API 연결 설정을 수정하세요.")
        return domain, None
    return domain, client


def render_overview(client: ApiClient, domain: str) -> None:
    product_id = st.session_state.get("detail_product")
    if product_id:
        render_detail(client, product_id)
        return
    try:
        products = client.products(domain).get("products", [])
    except ApiError as exc:
        report_api_error(exc)
        return
    st.subheader("리뷰로 살펴보는 상품")
    st.caption("상품별 리뷰와 ACOS 쿼드를 확인하고, 원하는 특성으로 추천을 받아보세요.")
    review_count = sum(p.get("review_count", 0) for p in products)
    quad_count = sum(p.get("quad_count", 0) for p in products)
    positive = sum(p.get("positive_ratio", 0) * p.get("quad_count", 0) for p in products)
    cols = st.columns(4)
    cols[0].metric("등록 상품", len(products))
    cols[1].metric("리뷰", review_count)
    cols[2].metric("ACOS 쿼드", quad_count)
    cols[3].metric("긍정 쿼드 비율", f"{positive / quad_count:.0%}" if quad_count else "—")
    if any(p.get("is_demo") for p in products):
        st.info("‘합성 예시’ 상품은 화면과 추천 동작을 보여주기 위한 데이터입니다. 실제 상품 평가나 구매 근거로 사용하지 마세요.")
    search = st.text_input("상품 검색", placeholder="상품명 또는 ID로 검색", key="catalog_search", max_chars=100)
    visible = [p for p in products if search.casefold() in (p.get("name", "") + " " + p.get("product_id", "")).casefold()]
    st.caption(f"{len(visible)}개 상품 표시")
    if not visible:
        st.info("표시할 상품이 없습니다. 검색어를 바꾸거나 데이터를 가져오세요.")
    for offset in range(0, len(visible), 3):
        columns = st.columns(3)
        for col, product in zip(columns, visible[offset:offset + 3]):
            with col, st.container(border=True):
                st.caption("합성 예시" if product.get("is_demo") else "가져온 데이터")
                st.subheader(literal_markdown(product.get("name", product["product_id"])), divider=False)
                st.caption(product["product_id"])
                left, right = st.columns(2)
                left.metric("리뷰", product.get("review_count", 0))
                right.metric("긍정 비율", f"{product.get('positive_ratio', 0):.0%}" if product.get("quad_count", 0) else "—")
                st.caption(f"{product.get('quad_count', 0)}개 쿼드 · {DOMAIN_LABELS.get(product.get('domain'), domain)}")
                st.button("리뷰와 근거 보기 →", key="catalog_open_" + product["product_id"],
                          on_click=open_product, args=(product["product_id"],), width="stretch")


def render_detail(client: ApiClient, product_id: str) -> None:
    st.button("← 상품 목록", on_click=close_product)
    try:
        detail = client.product(product_id)
    except ApiError as exc:
        report_api_error(exc)
        return
    product = detail.get("product", {})
    st.subheader(literal_markdown(product.get("name", product_id)))
    st.caption(f"상품 ID: {product_id}")
    if product.get("is_demo"):
        st.info("이 상품과 리뷰는 합성 예시입니다.")
    cols = st.columns(3)
    cols[0].metric("리뷰", product.get("review_count", 0))
    cols[1].metric("쿼드", product.get("quad_count", 0))
    cols[2].metric("긍정 쿼드 비율", f"{product.get('positive_ratio', 0):.0%}" if product.get("quad_count", 0) else "—")
    features = detail.get("features", [])
    if features:
        st.markdown("#### 관찰된 특성")
        st.dataframe([{"카테고리": category_label(f.get("category", "")), "특성": f.get("feature", ""),
                       "근거 수": f.get("count", 0), "긍정": f.get("pos", 0),
                       "중립": f.get("neu", 0), "부정": f.get("neg", 0), "점수": f.get("score", 0)}
                      for f in features], width="stretch", hide_index=True)
    st.markdown("#### 리뷰 원문과 ACOS")
    reviews = detail.get("reviews", [])
    if not reviews:
        st.info("등록된 리뷰가 없습니다.")
    for index, review in enumerate(reviews, 1):
        with st.container(border=True):
            st.caption(f"REVIEW {index:02d} · {review.get('review_id', '')}")
            st.text(review.get("text", ""))
            if review.get("quads"):
                st.dataframe(quad_rows(review["quads"]), width="stretch", hide_index=True)
            else:
                st.caption("이 리뷰에는 등록된 ACOS 주석이 없습니다.")
    json_download("상품 상세 JSON 저장", detail, "acos-product-detail.json", "detail_download")


def render_recommendations(client: ApiClient, domain: str) -> None:
    st.subheader("내가 원하는 특성으로 추천")
    st.caption("카테고리별 특성을 고르면 백엔드가 일치하는 리뷰 근거를 모아 상품 순위를 계산합니다.")
    try:
        categories = client.categories(domain).get("categories", [])
    except ApiError as exc:
        report_api_error(exc)
        return
    if not categories:
        st.info("선택 가능한 특성이 없습니다. ACOS 주석이 포함된 리뷰를 가져오세요.")
        return
    with st.form("recommendation_form"):
        selections = {}
        for offset in range(0, len(categories), 2):
            columns = st.columns(2)
            for col, entry in zip(columns, categories[offset:offset + 2]):
                category = entry["category"]
                counts = {f["feature"]: f.get("count", 0) for f in entry.get("features", [])}
                with col:
                    chosen = st.selectbox(category_label(category), [None] + list(counts),
                                          format_func=lambda value, c=counts: "선택하지 않음" if value is None else f"{value} · 근거 {c[value]}개",
                                          key="selection_" + category)
                if chosen is not None:
                    selections[category] = chosen
        left, right = st.columns(2)
        mode = left.radio("조건 결합", ["OR", "AND"], horizontal=True,
                          format_func=lambda value: "OR · 하나 이상 일치" if value == "OR" else "AND · 모두 일치",
                          key="recommendation_mode")
        top_k = right.slider("추천 상품 수", min_value=1, max_value=50, value=6, key="recommendation_top_k")
        submitted = st.form_submit_button("추천 결과 보기", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("recommendation_result", None)
        if not selections:
            st.warning("특성을 하나 이상 선택하세요.")
        else:
            try:
                with st.spinner("선택 조건과 리뷰 근거를 비교하고 있습니다…"):
                    result = client.recommend(domain, selections, mode, top_k)
                st.session_state["recommendation_result"] = {"domain": domain, "selections": selections,
                                                              "mode": mode, "response": result}
            except ApiError as exc:
                report_api_error(exc)
    stored = st.session_state.get("recommendation_result")
    if not stored:
        st.info("관심 있는 특성을 선택하고 추천을 실행하세요.")
        return
    st.divider()
    st.markdown("#### 추천 결과")
    st.caption(literal_markdown("요청 조건: " + " / ".join(f"{category_label(c)} = {f}" for c, f in stored["selections"].items()) + f" · {stored['mode']}"))
    st.caption("점수 = 일치한 쿼드의 (긍정 수 − 부정 수) ÷ 전체 수. 범위 −1~+1이며 구매 확률이나 모델 신뢰도가 아닙니다.")
    items = stored["response"].get("items", [])
    if not items:
        st.info("조건에 맞는 상품이 없습니다. 특성을 바꾸거나 OR 조건으로 다시 시도하세요.")
    for rank, item in enumerate(items, 1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.subheader(literal_markdown(f"{rank:02d} · {item.get('name', item.get('product_id', ''))}"))
                if item.get("is_demo"):
                    st.caption("합성 예시 상품")
                st.caption("일치 카테고리: " + ", ".join(category_label(c) for c in item.get("matched_categories", [])))
            right.metric("근거 점수", f"{item.get('score', 0):+.2f}")
            st.caption(f"리뷰 {item.get('review_count', 0)}개 · 상품 ID {item.get('product_id', '')}")
            with st.expander("추천에 사용된 리뷰 근거"):
                evidence = item.get("evidence", [])
                if not evidence:
                    st.caption("상세 근거가 없습니다.")
                for entry in evidence:
                    st.caption(literal_markdown(f"{category_label(entry.get('category', ''))} · {entry.get('feature', '')} · {SENTIMENT_LABELS.get(entry.get('sentiment'), entry.get('sentiment', ''))}"))
                    st.text(entry.get("text", ""))
                    st.dataframe(quad_rows([entry]), width="stretch", hide_index=True)
            st.button("상품 상세 보기 →", key="recommendation_open_" + item["product_id"],
                      on_click=open_product, args=(item["product_id"],))
    json_download("추천 결과 JSON 저장", stored, "acos-recommendations-" + domain + ".json", "recommendation_download")


def render_analysis(client: ApiClient, domain: str, analysis_mode: str, model_loaded: bool) -> None:
    st.subheader("리뷰 한 문장에서 네 가지 정보 찾기")
    st.caption("Aspect(대상), Category(범주), Opinion(의견), Sentiment(감성)을 백엔드에서 추출합니다.")
    unavailable = analysis_mode == "model" and not model_loaded
    if unavailable:
        st.error("학습 모델 모드로 설정되어 있지만 모델을 불러오지 못했습니다. 백엔드의 모델 설정을 확인하세요.")
    elif analysis_mode != "model":
        st.warning("현재는 입력에 따라 동작하는 키워드 규칙 기준선입니다. 학습된 ACOS 모델 추론 결과가 아니며, 문맥·부정·암시적 표현을 놓칠 수 있습니다.")
    examples = {"laptop": "The battery life is amazing but the screen is dull.",
                "rest16": "The food is delicious but the service is slow."}
    with st.form("analysis_form"):
        sentence = st.text_area("분석할 리뷰", value=examples[domain], height=140,
                                max_chars=8000, key="analysis_sentence")
        if analysis_mode != "model":
            st.caption("현재 규칙 기준선은 영어 키워드 중심입니다. 지원하지 않는 표현에는 결과가 없을 수 있습니다.")
        submitted = st.form_submit_button("리뷰 분석하기", type="primary", disabled=unavailable)
    if submitted:
        st.session_state.pop("analysis_result", None)
        if not sentence.strip():
            st.warning("분석할 리뷰를 입력하세요.")
        else:
            try:
                with st.spinner("백엔드에서 리뷰를 분석하고 있습니다…"):
                    st.session_state["analysis_result"] = client.analyze(sentence.strip(), domain)
            except ApiError as exc:
                report_api_error(exc)
    result = st.session_state.get("analysis_result")
    if not result:
        return
    st.divider()
    st.markdown("#### 분석 결과")
    st.caption("분석 방식: " + ("학습 모델" if result.get("analysis_mode") == "model" else "규칙 기반 기준선"))
    st.text(result.get("input_sentence", ""))
    quads = result.get("results", [])
    if quads:
        st.dataframe(quad_rows(quads), width="stretch", hide_index=True)
        st.caption(f"{len(quads)}개 쿼드 추출 · 분석 결과는 상품 데이터에 자동 저장되지 않습니다.")
    else:
        st.info("추출된 쿼드가 없습니다. 대상과 의견이 명확한 문장으로 다시 시도하세요.")
    if result.get("notice"):
        st.caption(result["notice"])
    json_download("분석 결과 JSON 저장", result, "acos-analysis-" + domain + ".json", "analysis_download")


def render_import(client: ApiClient, domain: str, metadata: dict) -> None:
    st.subheader("내 리뷰 데이터 가져오기")
    st.caption("파일은 먼저 미리보기로 열립니다. ‘백엔드에 가져오기’를 눌러야 상품과 리뷰가 저장됩니다.")
    body_limit = metadata.get("limits", {}).get("body_bytes", 2_000_000)
    st.info(f"현재 도메인: {DOMAIN_LABELS[domain]} · 파일 최대 5 MiB / 리뷰 1,000개 · API 전송 {body_limit:,} bytes 이하")
    sample = {"domain": domain, "records": [{"product_id": "my-product-01", "product_name": "내 상품",
              "text": "The battery is great." if domain == "laptop" else "The food is great.",
              "quads": [{"aspect": "battery" if domain == "laptop" else "food",
                         "category": "BATTERY#OPERATION_PERFORMANCE" if domain == "laptop" else "FOOD#QUALITY",
                         "opinion": "great", "sentiment": "Positive"}]}]}
    with st.expander("지원 형식과 예시"):
        st.caption("JSON: 레코드 배열 또는 domain, records 객체. TSV: product_id, product_name(선택), text, quads 헤더와 JSON 배열 형태의 quads 열.")
        st.caption("공식 ACOS TSV도 읽을 수 있습니다. 상품 ID가 없는 경우 각 문장은 별도 상품으로 처리됩니다. 같은 상품의 여러 리뷰에는 동일한 product_id를 사용하세요.")
        st.caption("빈 quads 배열은 미주석 리뷰로 저장되며 자동 분석하지 않습니다. 중복 레코드 처리와 최종 필드 검증은 백엔드가 수행합니다.")
        st.json(sample)
        json_download("가져오기 JSON 예시 저장", sample, "acos-import-example-" + domain + ".json", "import_example_download")
    uploaded = st.file_uploader("리뷰 데이터 선택", type=["json", "tsv"], key="import_file")
    if uploaded is None:
        return
    try:
        records, notices = parse_upload(uploaded.getvalue(), uploaded.name, domain)
    except ImportFormatError as exc:
        st.error(literal_markdown(str(exc)))
        return
    for notice in notices:
        st.warning(notice)
    payload_bytes = serialized_request_size(domain, records)
    st.caption(f"정규화된 API 요청 크기: {payload_bytes:,} / {body_limit:,} bytes")
    over_limit = payload_bytes > body_limit
    if over_limit:
        st.error("JSON으로 변환한 전송 내용이 백엔드 용량 제한을 초과합니다. 파일을 나누거나 레코드를 줄여 다시 선택하세요.")
    cols = st.columns(3)
    cols[0].metric("상품 ID", len({r["product_id"] for r in records}))
    cols[1].metric("리뷰 레코드", len(records))
    cols[2].metric("쿼드", sum(len(r["quads"]) for r in records))
    st.caption("첫 20개 레코드 미리보기")
    st.dataframe([{"상품 ID": r["product_id"], "상품명": r["product_name"], "리뷰": r["text"],
                   "쿼드 수": len(r["quads"])} for r in records[:20]], width="stretch", hide_index=True)
    with st.expander("첫 레코드의 실제 전송 내용"):
        st.json(records[0])
    if st.button("백엔드에 가져오기", type="primary", key="import_submit", disabled=over_limit):
        try:
            with st.spinner("검증한 레코드를 백엔드에 저장하고 있습니다…"):
                result = client.import_records(domain, records)
            st.session_state.pop("recommendation_result", None)
            close_product()
            st.success("가져오기를 완료했습니다. ‘상품 탐색’에서 데이터를 확인하세요.")
            st.json(result)
        except ApiError as exc:
            report_api_error(exc)


def main() -> None:
    st.set_page_config(page_title="ACOS Studio · 리뷰 기반 추천", page_icon="◈", layout="wide",
                       initial_sidebar_state="expanded")
    stylesheet = APP_ROOT / "assets" / "style.css"
    if stylesheet.exists():
        # Only static, repository-owned content is rendered as HTML.
        st.markdown("<style>" + stylesheet.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
    st.markdown('''<div class="acos-hero"><div><div class="acos-eyebrow">2025 GRADUATION PROJECT · REVIEW INTELLIGENCE</div>
<h1>리뷰 속 의견을,<br>선택의 근거로.</h1><p>ACOS로 읽고, 특성으로 비교하고, 근거와 함께 추천받으세요.</p></div>
<div class="acos-mark" aria-hidden="true"><span>A</span><span>C</span><span>O</span><span>S</span></div></div>''', unsafe_allow_html=True)
    domain, client = sidebar()
    if client is None:
        return
    try:
        health = client.health()
        metadata = client.meta()
    except ApiError as exc:
        report_api_error(exc)
        st.info("프로젝트 루트에서 python run.py를 실행하면 백엔드와 화면을 함께 시작할 수 있습니다.")
        return
    analysis_mode = metadata.get("analysis_mode", health.get("analysis_mode", "heuristic"))
    actual_model = bool(health.get("model_loaded")) and analysis_mode == "model"
    status_cols = st.columns([3, 2])
    status_cols[0].caption(f"● API 연결됨 · {DOMAIN_LABELS[domain]} 도메인")
    engine_label = "학습 모델" if actual_model else "학습 모델 연결 실패" if analysis_mode == "model" else "규칙 기반 기준선 (학습 모델 미연결)"
    status_cols[1].caption("분석 엔진 · " + engine_label)
    view = st.radio("화면", VIEWS, horizontal=True, key="app_view", label_visibility="collapsed",
                    on_change=close_product)
    st.divider()
    if view == "상품 탐색":
        render_overview(client, domain)
    elif view == "조건 추천":
        render_recommendations(client, domain)
    elif view == "리뷰 분석":
        render_analysis(client, domain, analysis_mode, actual_model)
    else:
        render_import(client, domain, metadata)
    st.divider()
    st.caption("ACOS Studio · 2025 졸업 프로젝트 | 합성 예시는 실제 상품 데이터가 아닙니다. 추천 점수는 등록된 리뷰 근거의 집계입니다.")


if __name__ == "__main__":
    main()
