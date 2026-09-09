# Aspect Sentiment Application
ACOS quadruple extraction을 화면/API에 연결하기 위한 application prototype입니다. 자체 학습 모델과 upstream 연구 구현을 명확히 구분합니다.

## 구조
`acos-backend/app.py`는 Flask/CORS 기반 POST /analyze를 제공합니다. `acos-frontend/acos_reco/streamlit_app.py`는 Streamlit 화면이며, 특성 정규화·상품별 index·선택 category 기반 추천은 `acos_reco/src`에 있습니다. `acos-backend/ACOS-main`의 Extract-Classify 구현은 NUSTM upstream 연구 코드이며 이 프로젝트가 처음 개발한 모델이 아닙니다.

## 현재 동작
predictor가 연결되지 않은 상태에서는 고정 예시 quadruple을 반환합니다. 입력 문장에 대해 실제 추론한 결과나 측정된 정확도가 아닙니다. 모델 경로가 존재한다는 이유만으로 학습/추론 완료로 표시하지 않습니다.

## 실행과 검증
Flask와 flask-cors 설치 환경에서 `python acos-backend/app.py`를 실행합니다. 기본 포트는 5001입니다. 공개 네트워크에 debug server를 노출하지 마세요. API의 빈 요청 400 및 mock 응답 구조를 smoke test로 확인하며 실제 모델 benchmark는 재실행하지 않습니다.

## Attribution 및 데이터
NUSTM의 ACOS 논문/코드 귀속을 [ATTRIBUTION](ATTRIBUTION.md)에 명시합니다. 논문 점수는 내 application 성능이 아닙니다. 원시 review dataset과 pretrained weights는 게시 대상이 아닙니다. upstream 재배포 허용 범위가 확인되기 전 private으로 유지합니다.
