import os
import math
import urllib.parse
import html
from typing import Dict, List, Tuple, Any, DefaultDict
from collections import defaultdict, Counter
import requests

import streamlit as st

# 내부 모듈: 기존 구현을 그대로 사용합니다
# - src/loader: load_acos_tsv, to_rows
# - src/indexer: build_product_index
# - src/recommender: Recommender
# - src/types: DEFAULT_DOMAIN_CATS
from src.loader import load_acos_tsv, to_rows
from src.indexer import build_product_index, sentence_to_product_id
from src.recommender import Recommender
from src.types import DEFAULT_DOMAIN_CATS


# 페이지 설정 및 공용 유틸

st.set_page_config(page_title="ACOS 카테고리 특성 추천 데모", layout="wide")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(APP_ROOT, p)

# UI 표시용 카테고리 한글 라벨 (대표적인 것만, 나머지는 그대로 노출)
CAT_LABELS_KO = {
    # Restaurant 예시
    "FOOD#QUALITY": "음식/품질",
    "FOOD#STYLE_OPTIONS": "음식/스타일",
    "FOOD#PRICES": "음식/가격",
    "DRINKS#QUALITY": "음료/품질",
    "DRINKS#STYLE_OPTIONS": "음료/스타일",
    "AMBIENCE#GENERAL": "분위기",
    "SERVICE#GENERAL": "서비스",
    "LOCATION#GENERAL": "위치",
    "RESTAURANT#GENERAL": "식당/종합",
    # Laptop 예시
    "LAPTOP#PRICE": "가격",
    "LAPTOP#GENERAL": "노트북/종합",
    "BATTERY#OPERATION_PERFORMANCE": "배터리/성능",
    "BATTERY#QUALITY": "배터리/품질",
    "CPU#OPERATION_PERFORMANCE": "CPU/성능",
    "CPU#QUALITY": "CPU/품질",
    "DISPLAY#QUALITY": "디스플레이/품질",
    "DISPLAY#USABILITY": "디스플레이/사용성",
    "GRAPHICS#OPERATION_PERFORMANCE": "그래픽/성능",
    "HARDWARE#QUALITY": "하드웨어/품질",
    "MOTHERBOARD#QUALITY": "메인보드/품질",
    "OPTICAL_DRIVES#QUALITY": "ODD/품질",
    "OS#USABILITY": "OS/사용성",
    "PORTS#USABILITY": "포트/사용성",
    "MEMORY#OPERATION_PERFORMANCE": "메모리/성능",
    "MEMORY#QUALITY": "메모리/품질",
    "KEYBOARD#USABILITY": "키보드/사용성",
    "MOUSE#USABILITY": "마우스/사용성",
    "TOUCHPAD#USABILITY": "터치패드/사용성",
    "HARD_DISC#QUALITY": "HDD/품질",
    "HARD_DISC#OPERATION_PERFORMANCE": "HDD/성능",
    "COMPANY#DESIGN_FEATURES": "회사/디자인",
}

def cat_label(cat: str, lang: str) -> str:
    if lang == "KR":
        return CAT_LABELS_KO.get(cat, cat)
    return cat  # EN 기본

# 특성(오피니언 텀) 정규화: 가벼운 규칙 기반으로 보기 좋게 만듭니다.
STOPWORDS = {
    "the","a","an","is","are","was","were","very","really","quite","just","so",
    "and","or","but","with","of","for","to","in","on","at","by","from","it","they",
}

PUNCTS = """!?,.;:\"'`()[]{}<>/\\|-_~*^`+="""

def normalize_feature(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    # 소문자화
    s = s.lower()
    # 구두점 제거(하이픈은 유지)
    s = "".join(ch for ch in s if ch not in PUNCTS or ch == '-')
    # 토큰화 및 불용어 제거
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    if not toks:
        toks = s.split()
    # 단순 어간 축약: 복수/동사어미
    norm = []
    for t in toks:
        if t.endswith('ing') and len(t) > 5:
            t = t[:-3]
        elif t.endswith('ers') and len(t) > 4:
            t = t[:-3]
        elif t.endswith('es') and len(t) > 3:
            t = t[:-2]
        elif t.endswith('s') and len(t) > 3:
            t = t[:-1]
        norm.append(t)
    # 상한 길이 3단어로 축약
    norm = norm[:3]
    # Title Case로 보기 좋게
    out = " ".join(norm).strip().title()
    # 빈 문자열 방지
    return out or text.strip().title()

# 그리드 UI 도우미
def grid(items: List[Any], cols: int = 3):
    rows = math.ceil(len(items) / cols)
    for r in range(rows):
        yield st.columns(cols)

# 제품 ID 제외 추천 필터
def exclude_current(results: List[Dict[str, Any]], current_pid: str) -> List[Dict[str, Any]]:
    return [r for r in results if r.get('product_id') != current_pid]


# 데이터 경로: 항상 ACOS 공식 데이터

CANDIDATE_BASES = [
    # 1) 프로젝트 내부에 복사해둔 경우
    os.path.join(APP_ROOT, "data", "ACOS", "data"),
    # 2) 워크스페이스 상위에 ACOS 레포가 형제 폴더로 있는 경우
    os.path.abspath(os.path.join(APP_ROOT, "..", "ACOS", "data")),
    # 3) 사용자 Documents/졸프/ACOS 경로
    os.path.expanduser("~/Documents/acos-backend/ACOS/data"),
    
]

DOM2DIR = {"rest16": "Restaurant-ACOS", "laptop": "Laptop-ACOS"}

def find_acos_tsv(domain: str, split: str) -> str:
    subdir = DOM2DIR.get(domain, "Restaurant-ACOS")
    filename = f"{domain}_quad_{split}.tsv"
    for base in CANDIDATE_BASES:
        cand = os.path.join(base, subdir, filename)
        if os.path.exists(cand):
            return cand
    return ""


# 로딩 및 인덱싱

@st.cache_data(show_spinner=False)
def load_and_index(tsv_path: str):
    quads = load_acos_tsv(tsv_path)
    rows = to_rows(quads)
    # 정규화된 표시용 특성 필드 추가
    for r in rows:
        r["feature_disp"] = normalize_feature(r.get("feature", ""))
        # 제품 ID 주입(인덱서와 동일 규칙)
        r["product_id"] = sentence_to_product_id(r.get("text", ""))
    agg_rows = build_product_index(rows)
    return rows, agg_rows

# 선택 가능한 카테고리별 대표 특성 추출
def top_features_by_category(rows: List[Dict[str, Any]], domain: str, topn: int = 12) -> Dict[str, List[str]]:
    cats = DEFAULT_DOMAIN_CATS.get(domain, [])
    out: Dict[str, List[str]] = {}
    for c in cats:
        cnt = Counter(r["feature_disp"] for r in rows if r.get("category") == c and r.get("feature_disp"))
        out[c] = [f for f, _ in cnt.most_common(topn)]
    return out

# 제품별 카테고리→특성 목록 추출 (상세 페이지)
def product_features(rows: List[Dict[str, Any]], product_id: str) -> Dict[str, List[str]]:
    pf: DefaultDict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r.get("product_id") == product_id:
            cat = r.get("category")
            feat = r.get("feature_disp")
            if cat and feat:
                pf[cat][feat] += 1
    return {c: [f for f, _ in cnt.most_common(12)] for c, cnt in pf.items()}


# 사이드바: 항상 공식 데이터 사용

def sidebar():
    st.sidebar.header("데이터 설정")
    # 쿼리 파라미터로 이전 상태 복원
    qp = st.query_params if hasattr(st, 'query_params') else {}
    def qp_get(name: str, default: str):
        v = qp.get(name)
        if isinstance(v, list):
            return v[0]
        return v if v is not None else default
    domain_default = qp_get("domain", "rest16")
    split_default = qp_get("split", "test")
    ui_mode_default = qp_get("uimode", "Dataset Browse")
    source_default = qp_get("source", "Gold (Official)")
    mode_default = qp_get("match", "OR")
    topk_default = int(qp_get("topk", "12")) if qp_get("topk", "").isdigit() else 12
    label_lang_default = qp_get("lang", "EN")

    domain = st.sidebar.selectbox("도메인", ["rest16", "laptop"], index=["rest16","laptop"].index(domain_default) if domain_default in ["rest16","laptop"] else 0, key="sb-domain")
    split_opts = ["train", "dev", "test"]
    split = st.sidebar.selectbox("스플릿", split_opts, index=split_opts.index(split_default) if split_default in split_opts else 2, key="sb-split")
    ui_mode = st.sidebar.radio("모드", ["Dataset Browse", "Live JSON Analysis"], index=0 if ui_mode_default not in ["Dataset Browse","Live JSON Analysis"] else ["Dataset Browse","Live JSON Analysis"].index(ui_mode_default), key="sb-uimode")
    source = st.sidebar.radio("데이터 소스", ["Gold (Official)", "Model (Predicted)"], index=0 if source_default not in ["Gold (Official)","Model (Predicted)"] else ["Gold (Official)","Model (Predicted)"].index(source_default), horizontal=True, key="sb-source")
    data_path = ""
    uploaded_file = None
    if source.startswith("Gold"):
        data_path = find_acos_tsv(domain, split)
        st.sidebar.text_input("공식 TSV 경로", value=data_path or "<찾는 중…>", disabled=True)
    else:
        st.sidebar.caption("모델 예측 TSV를 업로드하거나 경로를 직접 입력하세요. 포맷은 공식 TSV와 동일해야 합니다.")
        uploaded_file = st.sidebar.file_uploader("예측 TSV 업로드", type=["tsv"], accept_multiple_files=False)
        data_path = st.sidebar.text_input("예측 TSV 경로 (선택)", value="")
        if uploaded_file is not None:
            # 업로드 파일을 임시 경로에 저장
            tmp_dir = os.path.join(APP_ROOT, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"pred_{domain}_{split}.tsv")
            with open(tmp_path, "wb") as wf:
                wf.write(uploaded_file.getbuffer())
            data_path = tmp_path

    st.sidebar.header("추천 설정")
    # 기본을 OR로 변경 (더 많은 후보 노출 후 점진적 필터링 UX)
    mode = st.sidebar.radio("매칭 정책", ["AND", "OR"], index= ["AND","OR"].index(mode_default) if mode_default in ["AND","OR"] else 1, horizontal=True, key="sb-match")
    topk = st.sidebar.slider("Top-K", min_value=5, max_value=50, value=topk_default if 5 <= topk_default <= 50 else 12, step=1, key="sb-topk")
    label_lang = st.sidebar.radio("라벨 언어", ["EN", "KR"], index=["EN","KR"].index(label_lang_default) if label_lang_default in ["EN","KR"] else 0, horizontal=True, key="sb-lang")

    # 현재 상태를 쿼리 파라미터에 반영 (pid는 유지)
    params = st.query_params if hasattr(st, 'query_params') else {}
    params["domain"] = domain
    params["split"] = split
    params["uimode"] = ui_mode
    params["source"] = source
    params["match"] = mode
    params["topk"] = str(topk)
    params["lang"] = label_lang
    # pid는 detail 페이지 접근 시 이미 존재할 수 있음 -> 그대로 두되 변경 없음
    st.query_params.update(params)

    return domain, split, data_path, source, mode, topk, label_lang, ui_mode


# 렌더링

CARD_CSS = """
<style>
.card {
  border: 1px solid #E5E7EB; border-radius: 10px; padding: 14px; background: #fff;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; margin-right: 6px; background: #F3F4F6; }
.cat { background: #EEF2FF; color: #3730A3; }
.feat { background: #ECFDF5; color: #065F46; }
.kpi { color: #111827; font-weight: 600; }
.btn-inline { display:inline-block; margin-left: 8px; }
.section-title { font-weight:700; font-size:16px; margin: 4px 0 8px; }
.hr { height:1px; background:#F3F4F6; border:0; margin:12px 0; }
</style>
<style>
/* 추가: ACOS 쿼드 표시 스타일 */
.quad-wrapper { margin-top:6px; background:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px; padding:6px 8px; }
.quad-title { font-size:11px; font-weight:600; color:#374151; margin-bottom:2px; display:flex; align-items:center; justify-content:space-between; }
.quad-line { font-size:11px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height:1.3; white-space:pre-wrap; word-break:break-word; }
.quad-toggle-btn { background:#FFF; border:1px solid #D1D5DB; color:#374151; font-size:10px; padding:2px 6px; border-radius:4px; cursor:pointer; }
.quad-toggle-btn:hover { background:#F3F4F6; }
.legend { display:flex; gap:6px; flex-wrap:wrap; margin-top:4px; }
.legend span { font-size:10px; padding:2px 4px; border-radius:4px; border:1px solid #D1D5DB; background:#fff; }
.token-a { background:#FEF3C7; padding:2px 4px; border-radius:4px; }
.token-o { background:#E0F2FE; padding:2px 4px; border-radius:4px; }
.token-a.token-o { background:#FDE68A; }
.sent-pos { color:#065F46; font-weight:600; }
.sent-neu { color:#374151; font-weight:600; }
.sent-neg { color:#B91C1C; font-weight:600; }
.sent-pos-uline { box-shadow: inset 0 -2px 0 #059669; }
.sent-neu-uline { box-shadow: inset 0 -2px 0 #6B7280; }
.sent-neg-uline { box-shadow: inset 0 -2px 0 #DC2626; }
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

# 결과 카드 렌더링 (quads_map 전달 시 ACOS 모든 쿼드 표시)
def render_result_card(item: Dict[str, Any], label_lang: str = "EN", quads_map: Dict[str, List[Dict[str, Any]]] | None = None):
    cat = item.get("category")
    cat_disp = cat_label(cat, label_lang)
    feat_disp = item.get("feature_disp", "")
    # score 숨김: 필요시 재활성화용으로 값만 유지
    _score_hidden = item.get("final_score", item.get("score", 0.0))
    pid = item.get("product_id", "")
    text = item.get("text", "").strip()

    with st.container(border=False):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([0.75, 0.25])
        with left:
            st.markdown(f"<span class='badge cat'>{cat_disp}</span> <span class='badge feat'>{feat_disp}</span>", unsafe_allow_html=True)
            # 하이라이트 준비
            highlighted_html = html.escape(text)
            if quads_map is not None and pid in quads_map and text:
                # 토큰 기준 재구성
                tokens = text.split()
                # 각 토큰에 여러 태그 가능 (aspect/opinion)
                mark_types = [[] for _ in tokens]
                # sentiment 집계용 (토큰 인덱스 -> 카운트)
                senti_counter = [Counter() for _ in tokens]
                for q in quads_map[pid]:

                    a_txt = (q.get("aspect_text") or "").strip()
                    o_txt = (q.get("opinion_text") or "").strip()
                    s_val = q.get("sentiment")
                    def mark_phrase(p_txt: str, tag: str):
                        if not p_txt:
                            return
                        p_tokens = p_txt.split()
                        L = len(p_tokens)
                        if L == 0:
                            return
                        for i in range(len(tokens)-L+1):
                            if [t.lower() for t in tokens[i:i+L]] == [t.lower() for t in p_tokens]:
                                for k in range(i, i+L):
                                    if tag not in mark_types[k]:
                                        mark_types[k].append(tag)
                                        # sentiment 기록 (A/O 토큰에만 반영)
                                        if isinstance(s_val, int):
                                            senti_counter[k][s_val] += 1
                    mark_phrase(a_txt, "A")
                    mark_phrase(o_txt, "O")
                # 재조합
                out_tokens = []
                for idx, tok in enumerate(tokens):
                    esc = html.escape(tok)
                    if not mark_types[idx]:
                        out_tokens.append(esc)
                    else:
                        cls = []
                        if "A" in mark_types[idx] and "O" in mark_types[idx]:
                            cls.append("token-a token-o")
                        elif "A" in mark_types[idx]:
                            cls.append("token-a")
                        elif "O" in mark_types[idx]:
                            cls.append("token-o")
                        # sentiment underline
                        if senti_counter[idx]:
                            most = senti_counter[idx].most_common(1)[0][0]
                            if most == 2:
                                cls.append("sent-pos-uline")
                            elif most == 1:
                                cls.append("sent-neu-uline")
                            else:
                                cls.append("sent-neg-uline")
                        out_tokens.append(f"<span class='{' '.join(cls)}'>{esc}</span>")
                highlighted_html = " ".join(out_tokens)
            st.markdown(f"<div style='margin-top:6px; line-height:1.5'>{highlighted_html}</div>", unsafe_allow_html=True)
            if quads_map is not None:
                pid_quads = quads_map.get(pid, []) if quads_map else []
                if pid_quads:
                    # 전체 표시 (토글 적용)
                    exp_key = f"exp-{pid}"
                    expanded = st.session_state.get(exp_key, False)
                    toggle_label = "접기" if expanded else "펼치기"
                    # 토글 버튼 + legend (중첩 columns 회피: flex 컨테이너 사용)
                    # 체크박스로 전환 (단일 클릭 신뢰성 향상)
                    expanded = st.checkbox("ACOS Quads", value=expanded, key=f"chk-{exp_key}", help="이 상품의 모든 쿼드 펼치기/접기")
                    st.session_state[exp_key] = expanded
                    st.markdown("<div class='legend'><span style='background:#FEF3C7'>A</span><span style='background:#E0F2FE'>O</span><span style='background:#FDE68A'>A+O</span><span style='padding:2px 4px;border:1px solid #D1D5DB;'>S=밑줄색(Pos/Neu/Neg)</span></div>", unsafe_allow_html=True)
                    if expanded:
                        lines = []
                        for q in pid_quads:
                            a = q.get("aspect_text") or "-"
                            c2 = q.get("category") or "-"
                            o = q.get("opinion_text") or "-"
                            s_val = q.get("sentiment")
                            if isinstance(s_val, int):
                                if s_val == 2:
                                    s = "<span class='sent-pos'>POS</span>"
                                elif s_val == 1:
                                    s = "<span class='sent-neu'>NEU</span>"
                                else:
                                    s = "<span class='sent-neg'>NEG</span>"
                            else:
                                s = html.escape(str(s_val))
                            line = f"A:{html.escape(a)} | C:{html.escape(c2)} | O:{html.escape(o)} | S:{s}"
                            lines.append(line)
                        quad_html = "".join([f"<div class='quad-line'>{ln}</div>" for ln in lines])
                        st.markdown(f"<div class='quad-wrapper'><div class='quad-title'>ACOS Quads (A/C/O/S)</div>{quad_html}</div>", unsafe_allow_html=True)
        with right:
            safe_pid = html.escape(pid)
            # Product ID를 상세 링크로 (score 제거)
            params = st.query_params
            params["pid"] = pid
            detail_url = "?" + urllib.parse.urlencode(params)
            st.markdown(f"<div style='text-align:right;font-size:11px;'><a href='{detail_url}' style='text-decoration:none;color:#2563EB;font-weight:600;'>PID: {safe_pid}</a></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# 메인 추천 탭
def page_main(domain: str, split: str, data_path: str, source: str, mode: str, topk: int, label_lang: str):
    if not data_path or not os.path.exists(data_path):
        if source.startswith("Gold"):
            st.warning("공식 TSV 파일을 찾을 수 없습니다. ACOS 레포의 데이터 경로를 확인하세요.")
        else:
            st.warning("모델 예측 TSV를 업로드하거나 유효한 경로를 입력하세요.")
        st.stop()

    with st.spinner("데이터 로딩 및 인덱싱 중…"):
        rows, agg_rows = load_and_index(data_path)
        rec = Recommender(agg_rows)
        top_feats = top_features_by_category(rows, domain)

    st.subheader(f"카테고리별 특성 선택 · 소스: {'Gold' if source.startswith('Gold') else 'Model'}")
    selections: Dict[str, str] = {}
    cats = [c for c in DEFAULT_DOMAIN_CATS.get(domain, []) if top_feats.get(c)]

    # 3열 그리드에 선택 컴포넌트 배치
    for i in range(0, len(cats), 3):
        cols = st.columns(3)
        for j, c in enumerate(cats[i:i+3]):
            with cols[j]:
                label = cat_label(c, label_lang)
                base = top_feats[c]
                opts = ["(None)"] + base
                # 현재 선택 유지 (세션 상태 기반)
                prev = st.session_state.get(f"sel-{c}")
                if prev and prev in opts:
                    default_idx = opts.index(prev)
                else:
                    default_idx = 1 if len(opts) > 1 else 0
                sel = st.selectbox(label, options=opts, index=default_idx, key=f"sel-{c}")
                if sel != "(None)":
                    selections[c] = sel

    # 추천 실행
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("추천 결과")
    rec_rows = rec.recommend(selections=selections, mode=mode, top_k=topk)

    # 각 추천 항목에 대표 텍스트/특성 보완 주입 + 쿼드 수집
    sample_text_by_pid: Dict[str, str] = {}
    sample_feat_by_pid_cat: Dict[Tuple[str, str], str] = {}
    quads_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        pid = r.get("product_id")
        if pid and pid not in sample_text_by_pid:
            sample_text_by_pid[pid] = r.get("text", "").strip()
        c = r.get("category")
        fd = r.get("feature_disp", "")
        if pid and c and fd and (pid, c) not in sample_feat_by_pid_cat:
            sample_feat_by_pid_cat[(pid, c)] = fd
        if pid:
            quads_map[pid].append(r)
    for it in rec_rows:
        pid = it.get("product_id")
        it["text"] = sample_text_by_pid.get(pid, "")
        # feature_disp 비어 있으면 보완
        if not it.get("feature_disp"):
            cat = it.get("category")
            it["feature_disp"] = sample_feat_by_pid_cat.get((pid, cat), "")

    # 그리드 렌더링
    items = rec_rows
    cols = 3
    for row_idx, col_group in enumerate(grid(items, cols=cols)):
        for c_idx in range(cols):
            idx = row_idx * cols + c_idx
            if idx < len(items):
                with col_group[c_idx]:
                    render_result_card(items[idx], label_lang, quads_map)

# 상세 페이지: 선택한 제품 기반 특성 수정 → 새 추천
def page_detail(domain: str, split: str, data_path: str, source: str, mode: str, topk: int, pid: str, label_lang: str):
    if not data_path or not os.path.exists(data_path):
        if source.startswith("Gold"):
            st.warning("공식 TSV 파일을 찾을 수 없습니다. ACOS 레포의 데이터 경로를 확인하세요.")
        else:
            st.warning("모델 예측 TSV를 업로드하거나 유효한 경로를 입력하세요.")
        st.stop()

    with st.spinner("데이터 로딩 및 인덱싱 중…"):
        rows, agg_rows = load_and_index(data_path)
        rec = Recommender(agg_rows)

    # 현재 제품의 특성 모음
    pf = product_features(rows, pid)

    st.markdown(f"### 상품 상세 · 소스: {'Gold' if source.startswith('Gold') else 'Model'}")
    st.caption(f"Product ID: {pid}")

    # 해당 제품의 대표 문장 하나 노출
    samples = [r for r in rows if r.get("product_id") == pid]
    if samples:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(samples[0].get("text", "").strip())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("카테고리별 특성 조정")

    # 제품에 존재하지 않는 카테고리도 선택 가능하게: 전역 top 특성과 합침
    global_top = top_features_by_category(rows, domain)
    cats = DEFAULT_DOMAIN_CATS.get(domain, [])

    selections: Dict[str, str] = {}
    for i in range(0, len(cats), 3):
        cols = st.columns(3)
        for j, c in enumerate(cats[i:i+3]):
            with cols[j]:
                label = cat_label(c, label_lang)
                raw_opts = list(dict.fromkeys((pf.get(c, []) + global_top.get(c, []))))
                opts = ["(None)"] + (raw_opts if raw_opts else [])
                prev = st.session_state.get(f"detail-{c}")
                if prev and prev in opts:
                    default_idx = opts.index(prev)
                else:
                    default_idx = 1 if len(opts) > 1 else 0
                sel = st.selectbox(label, options=opts, index=default_idx, key=f"detail-{c}")
                if sel != "(None)":
                    selections[c] = sel

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("이 특성 조합으로 추천된 다른 상품")

    results = rec.recommend(selections=selections, mode=mode, top_k=topk)
    results = exclude_current(results, pid)

    # 추천 항목에 텍스트/특성 보완 주입 + 쿼드 수집
    sample_text_by_pid: Dict[str, str] = {}
    sample_feat_by_pid_cat: Dict[Tuple[str, str], str] = {}
    quads_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        p2 = r.get("product_id")
        if p2 and p2 not in sample_text_by_pid:
            sample_text_by_pid[p2] = r.get("text", "").strip()
        c = r.get("category")
        fd = r.get("feature_disp", "")
        if p2 and c and fd and (p2, c) not in sample_feat_by_pid_cat:
            sample_feat_by_pid_cat[(p2, c)] = fd
        if p2:
            quads_map[p2].append(r)
    for it in results:
        p2 = it.get("product_id")
        it["text"] = sample_text_by_pid.get(p2, "")
        if not it.get("feature_disp"):
            cat = it.get("category")
            it["feature_disp"] = sample_feat_by_pid_cat.get((p2, cat), "")

    if not results:
        st.info("해당 조합으로 추천 결과가 없습니다. 특성을 변경해 보세요.")
    else:
        # 그리드 렌더링
        cols = 3
        for row_idx, col_group in enumerate(grid(results, cols=cols)):
            for c_idx in range(cols):
                idx = row_idx * cols + c_idx
                if idx < len(results):
                    with col_group[c_idx]:
                        render_result_card(results[idx], label_lang, quads_map)

    # 상단으로 돌아가기
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    home_url = "?"  # 쿼리스트링 초기화
    st.markdown(f"<a href='{home_url}'>← 목록으로 돌아가기</a>", unsafe_allow_html=True)


# 엔트리포인트


# Live JSON Analysis (백엔드 즉시 결과 활용)


SENTI_MAP = {
    "positive": 2,
    "pos": 2,
    "negative": 0,
    "neg": 0,
    "neutral": 1,
    "neu": 1,
}

def parse_backend_json(raw: str) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    import json
    errors: List[str] = []
    if not raw.strip():
        return "", [], ["빈 JSON"]
    try:
        data = json.loads(raw)
    except Exception as e:
        return "", [], [f"JSON 파싱 실패: {e}"]
    if not isinstance(data, dict):
        return "", [], ["최상위 구조는 dict 여야 합니다."]
    sent = data.get("input_sentence", "").strip()
    res = data.get("results", [])
    if not sent:
        errors.append("input_sentence 누락")
    if not isinstance(res, list):
        errors.append("results 는 list 여야 합니다.")
        res = []
    rows: List[Dict[str, Any]] = []
    pid = f"live_{abs(hash(sent)) % 10**8}"
    for idx, r in enumerate(res):
        if not isinstance(r, dict):
            errors.append(f"results[{idx}] dict 아님")
            continue
        aspect = (r.get("aspect") or "").strip()
        category = (r.get("category") or "").strip()
        opinion = (r.get("opinion") or "").strip()
        senti_raw = (r.get("sentiment") or "").strip().lower()
        senti = SENTI_MAP.get(senti_raw, 1)  # default neutral
        if not category:
            errors.append(f"[{idx}] category 없음")
            continue
        feat_source = aspect or opinion or "implicit"
        feature = feat_source.lower()
        rows.append({
            "text": sent,
            "category": category,
            "sentiment": senti,
            "feature": feature,
            "aspect_text": aspect,
            "opinion_text": opinion,
            "product_id": pid,
            "feature_disp": normalize_feature(feature),
        })
    return sent, rows, errors

def page_live_json(domain: str, split: str, data_path: str, source: str, mode: str, topk: int, label_lang: str):
    st.subheader("Live Sentence Analysis (백엔드 연동)")

    # 우리가 만든 백엔드 Flask API의 주소
    API_URL = "http://127.0.0.1:5001/analyze"



    # 1. 문장을 입력받는 텍스트 입력창
    # st.session_state를 사용하여 사용자가 이전에 입력했던 문장을 기억하게 만듭니다.
    sentence_to_analyze = st.text_input(
        "분석할 리뷰 문장을 입력하세요:",
        value=st.session_state.get("_live_sent_input", "The battery life is amazing but the screen is dull.")
    )

    # 2. '분석 요청' 버튼과 '리셋' 버튼
    col_btn1, col_btn2, _ = st.columns([0.25, 0.25, 0.5])
    with col_btn1:
        run_analysis = st.button("백엔드에 분석 요청", type="primary")
    with col_btn2:
        reset_ctx = st.button("Reset Context")

    # 리셋 버튼 로직 (기존 코드와 동일)
    if reset_ctx:
        for k in list(st.session_state.keys()):
            if k.startswith("live-") or k.startswith("_live_json") or k.startswith("_live_sent_input"):
                del st.session_state[k]
        st.experimental_rerun()

    # '분석 요청' 버튼을 눌렀을 때 실행되는 로직
    if run_analysis:
        if not sentence_to_analyze.strip():
            st.warning("분석할 문장을 입력해주세요.")
        else:
            try:
                # 백엔드 API에 보낼 데이터 (JSON 형식)
                payload = {"sentence": sentence_to_analyze}

                # 백엔드 API 호출 (POST 요청) 및 스피너 표시
                with st.spinner("백엔드 모델 분석 중..."):
                    response = requests.post(API_URL, json=payload, timeout=30)
                    response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

                # 백엔드로부터 받은 JSON 응답 텍스트
                raw_json_response = response.text

                # 기존 parse_backend_json 함수를 재활용하여 받은 JSON을 파싱
                sent, live_rows, errors = parse_backend_json(raw_json_response)

                if errors:
                    for e in errors: st.error(e)
                if live_rows:
                    st.success("백엔드로부터 분석 결과를 성공적으로 받았습니다!")
                    # 세션 상태에 결과 저장 (기존 로직과 호환)
                    st.session_state["_live_json_rows"] = live_rows
                    st.session_state["_live_json_sent"] = sent
                    st.session_state["_live_json_raw"] = raw_json_response
                    st.session_state["_live_sent_input"] = sentence_to_analyze # 입력 문장도 기억

            except requests.exceptions.RequestException as e:
                st.error(f"백엔드 API 호출에 실패했습니다: {e}")
            except Exception as e:
                st.error(f"알 수 없는 오류가 발생했습니다: {e}")





    # 추천을 위한 컨텍스트 데이터 로드
    ctx_rows: List[Dict[str, Any]] = []
    agg_rows: List[Dict[str, Any]] = []
    if data_path and os.path.exists(data_path):
        with st.spinner("컨텍스트 데이터 로드 중..."):
            ctx_rows, agg_rows = load_and_index(data_path)
    else:
        st.warning("추천을 위한 컨텍스트 데이터(공식/예측 TSV)를 먼저 사이드바에서 설정하세요.")

    # 세션에 저장된 분석 결과가 있으면 화면에 추천 UI를 그립니다.
    live_rows = st.session_state.get("_live_json_rows", [])
    sent = st.session_state.get("_live_json_sent", "")

    if not live_rows:
        st.info("분석할 문장을 입력하고 '백엔드에 분석 요청' 버튼을 누르세요.")
        return

    # 추출 결과 표시
    st.markdown("#### 입력 문장")
    if sent:
        st.info(sent)
    st.markdown("#### 추출된 속성")
    cols = st.columns(3)
    for i, r in enumerate(live_rows):
        with cols[i % 3]:
            st.write(f"• {r['category']} : {r['feature_disp']} (sent={r['sentiment']})")

    # 카테고리별 특성 편집 및 추천 UI (기존 코드와 동일)
    extracted_cat2feat = {r['category']: r['feature_disp'] for r in live_rows if r.get('feature_disp')}
    top_ctx = top_features_by_category(ctx_rows, domain) if ctx_rows else {}
    all_cats = sorted(set(list(extracted_cat2feat.keys()) + list(top_ctx.keys())))
    st.markdown("#### 카테고리별 특성 편집")

    selections: Dict[str, str] = {}
    for i in range(0, len(all_cats), 3):
        ccols = st.columns(3)
        for j, c in enumerate(all_cats[i:i+3]):
            with ccols[j]:
                base_opts: List[str] = []
                if extracted_cat2feat.get(c):
                    base_opts.append(extracted_cat2feat[c])
                base_opts += [f for f in top_ctx.get(c, []) if f not in base_opts]
                final_opts = ["(None)"] + (base_opts if base_opts else [])
                label = cat_label(c, label_lang)
                prev = st.session_state.get(f"live-{c}")
                if prev and prev in final_opts:
                    default_idx = final_opts.index(prev)
                else:
                    default_idx = 1 if len(final_opts) > 1 else 0
                sel = st.selectbox(label, options=final_opts, index=default_idx, key=f"live-{c}")
                if sel != "(None)":
                    selections[c] = sel

    st.markdown("---")
    st.markdown("#### 추천 결과")
    if not agg_rows:
        st.warning("컨텍스트 데이터가 없어 추천 풀을 구성할 수 없습니다. TSV를 설정 후 다시 시도하세요.")
        return
    rec = Recommender(agg_rows)
    rec_rows = rec.recommend(selections=selections, mode=mode, top_k=topk)
    
    # 텍스트 매핑 및 쿼드 수집 (기존 코드와 동일)
    sample_text_by_pid: Dict[str, str] = {}
    quads_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in ctx_rows:
        pid = r.get("product_id")
        if pid and pid not in sample_text_by_pid:
            sample_text_by_pid[pid] = r.get("text", "").strip()
        if pid:
            quads_map[pid].append(r)
    for it in rec_rows:
        pid = it.get("product_id")
        it["text"] = sample_text_by_pid.get(pid, "")
        
    if not rec_rows:
        st.info("해당 선택으로 추천 결과가 없습니다. 특성을 변경해 보세요.")
    else:
        cols = 3
        for row_idx, col_group in enumerate(grid(rec_rows, cols=cols)):
            for c_idx in range(cols):
                idx = row_idx * cols + c_idx
                if idx < len(rec_rows):
                    with col_group[c_idx]:
                        render_result_card(rec_rows[idx], label_lang, quads_map)


def main():
    st.title("🛍️ ACOS 기반 카테고리·특성 추천 데모")
    domain, split, data_path, source, mode, topk, label_lang, ui_mode = sidebar()

    pid = st.query_params.get("pid", "") if hasattr(st, 'query_params') else ""
    if ui_mode == "Live JSON Analysis":
        page_live_json(domain, split, data_path, source, mode, topk, label_lang)
        return
    if pid:
        page_detail(domain, split, data_path, source, mode, topk, pid, label_lang)
    else:
        page_main(domain, split, data_path, source, mode, topk, label_lang)

if __name__ == "__main__":
    main()
