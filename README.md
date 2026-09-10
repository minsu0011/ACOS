# 리뷰의 속성과 감정을 연결한 추천 앱

리뷰에서 무엇을 어떤 감정으로 평가했는지 보여주고 관심 특성에 맞는 항목을 찾는 application prototype입니다. ACOS 연구의 출력 형식을 Flask API와 Streamlit 화면에 연결하는 구조를 다룹니다.

## 사용 기술

Python, Flask, flask-cors, Streamlit, Requests를 사용합니다. 연구 코드의 PyTorch·BERT 기반 Extract-Classify와 application의 API·화면을 구분합니다.

## 원 논문과 이 프로젝트의 범위

ACOS는 Aspect, Category, Opinion, Sentiment 네 요소를 추출하는 문제입니다. 긍정/부정 점수 하나보다 “어떤 속성의 무엇을 좋게 평가했는가”를 구조적으로 다루는 접근입니다.

NUSTM의 Extract-Classify 구현과 모델은 원 저자의 작업입니다. 여기서 연결한 부분은 요청/응답 형식, 리뷰 로딩, 특성 정규화, 상품별 index와 관심 category 기반 추천 화면입니다. [논문과 출처](ATTRIBUTION.md)에 역할을 구분했습니다.

## 개발 과정과 application 흐름

연구 출력의 quadruple을 화면에서 사용할 수 있는 행 구조로 다뤘습니다. 같은 뜻의 특성이 표기 차이로 나뉘지 않도록 소문자화·구두점·불용어 처리 같은 가벼운 정규화를 넣었습니다. 항목별 index를 만든 뒤 선택한 category와 특성의 빈도·감정을 추천에 연결했습니다.

```text
ACOS TSV → loader → 정규화·상품 index → recommender → Streamlit
입력 문장 → Flask /analyze → quadruple 응답
```

두 경로를 구분해야 합니다. 현재 Flask predictor는 연결되지 않아 고정 예시를 반환합니다. API 응답은 입력 문장에 대한 실제 모델 추론이 아닙니다. TSV 기반 탐색·추천과 모델 연동 완료는 다른 범위입니다.

## 코드 위치

- [acos-backend/app.py](acos-backend/app.py): 빈 입력 검사와 POST /analyze
- [acos-frontend/acos_reco](acos-frontend/acos_reco): 화면과 loader/indexer/recommender
- [acos-backend/ACOS-main](acos-backend/ACOS-main): upstream 연구 구현

## 실행

```bash
pip install flask flask-cors streamlit requests
python acos-backend/app.py
streamlit run acos-frontend/acos_reco/streamlit_app.py
```

Flask는 `127.0.0.1:5001`에서 동작합니다. 화면에는 사용 권한이 있는 ACOS TSV를 준비합니다. 실제 추론에는 별도 모델 환경·가중치와 predictor 연결이 필요합니다.

## 남은 부분

논문 benchmark 성능은 이 application의 성능이 아닙니다. 추천은 입력 리뷰와 정규화 규칙에 영향을 받으며 실제 사용자 만족도나 구매 성과로 평가한 시스템은 아닙니다.
