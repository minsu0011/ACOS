# 출처와 구현 범위

## ACOS 연구

Cai et al., ACL 2021, *Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions*:

- 논문: https://aclanthology.org/2021.acl-long.29/
- 원 저장소: https://github.com/NUSTM/ACOS

아래 두 디렉터리는 이전 프로젝트에서 보존한 upstream 기반 연구 코드입니다. 서로 일부 파일이 다르므로 이번 애플리케이션 정리에서 임의로 합치지 않았습니다.

- acos-backend/ACOS-main/
- acos-frontend/acos_reco/ACOS/

논문의 Extract-Classify 구현·모델과 원 데이터의 기여는 연구 저자에게 있습니다. 이 앱의 Flask API, SQLite 저장, Streamlit 화면, 데이터 가져오기·추천 흐름과 구분합니다. 논문 benchmark는 이 앱의 자체 실험 결과가 아닙니다.

## BERT 관련 파일

일부 BERT 파일은 Google AI Language Team, Hugging Face, NVIDIA의 저작권 및 Apache-2.0 고지를 가지고 있습니다. 기존 header를 유지했으며 해당 라이선스 본문을 third_party/Apache-2.0.txt에 넣었습니다.

이 고지가 ACOS 저장소 전체나 benchmark 데이터에 Apache 라이선스를 부여하는 것은 아닙니다. 모든 제3자 코드에 포괄적인 새 라이선스를 붙이지 않습니다. 원 저장소의 이용 조건과 파일별 고지를 확인해야 합니다.

## 애플리케이션과 예제

기존 Flask·Streamlit 구조 위에서 API 계약·저장·데이터 검증·규칙 기반 분석·추천·화면 연결을 정리했습니다. 기본 분석기는 pretrained 연구 모델이 아니며, 주어진 표현 규칙의 한계를 문서에 명시합니다.

데모 상품과 리뷰는 동작 확인을 위해 직접 작성한 합성 예제입니다. 실제 상품 평가·구매 성과를 나타내지 않습니다. 사용자 리뷰·DB·가중치는 현재 배포 파일에 포함하지 않습니다.
