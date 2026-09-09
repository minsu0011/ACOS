# ACOS 기반 카테고리 특성 추천 데모 (로컬)

- ACOS(Aspect-Category-Opinion-Sentiment) 데이터/모델의 결과를 그대로 활용해, 카테고리별 특성 1개씩 선택 → 해당 상품(아이템)을 추천하는 로컬 데모입니다.
- 로컬 실행용 초경량 스택: Streamlit + pandas (선택: DuckDB)

## 구조

- streamlit_app.py: UI/앱 엔트리
- src/
  - loader.py: ACOS TSV 로더 및 정규화
  - indexer.py: 상품 단위 집계 인덱스 생성
  - recommender.py: 필터/스코어링 로직
  - types.py: 공용 타입/상수
- data/
  - quads_sample.tsv: 샘플(레스토랑) 일부 라인
- assets/
  - style.css: 간단한 카드/배지 스타일

## 실행

1) 의존성 설치

```bash
pip install -r requirements.txt
```

2) 앱 실행

```bash
streamlit run streamlit_app.py
```

3) 데이터 경로
- 기본은 `data/quads_sample.tsv` 샘플을 사용합니다.
- 전체 데이터 사용 시, ACOS 레포의 `data/Restaurant-ACOS/rest16_quad_*.tsv` 또는 `data/Laptop-ACOS/laptop_quad_*.tsv`를 다운로드하여 경로를 지정하세요.

환경변수 또는 앱 사이드바에서 데이터 파일을 선택할 수 있습니다.

## 참고
- 감성 레이블: 0=negative, 1=neutral, 2=positive (ACOS 정의)
- 쿼드 포맷(각 라인):
  - "문장\t[a_start,a_end category sentiment o_start,o_end\t...]"
  - 인덱스가 -1,-1이면 implicit(암시적)임을 의미합니다.
