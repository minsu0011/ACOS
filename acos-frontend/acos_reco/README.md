# ACOS Studio — Streamlit 화면

2025 졸업 프로젝트의 리뷰 탐색·분석·추천 화면입니다. 기존 Streamlit 구조를 유지하며, 데이터 조회와 리뷰 분석 및 추천 점수 계산은 Flask 백엔드의 `/api/*`를 사용합니다. 로컬 TSV 파일이 없어도 백엔드의 합성 예시 데이터로 실행됩니다.

## 실행

프로젝트 루트의 `python run.py`로 백엔드와 화면을 함께 실행할 수 있습니다. 별도로 실행하려면 백엔드를 먼저 시작한 뒤 이 폴더에서 실행합니다.

```powershell
python -m pip install -r requirements.txt
$env:ACOS_API_URL = "http://127.0.0.1:5001"
$env:ACOS_API_TIMEOUT = "10"
python -m streamlit run streamlit_app.py
```

API 주소와 응답 제한 시간(1~60초)은 사이드바에서도 변경할 수 있습니다. 연결 오류는 화면에 표시되며 ‘다시 연결’로 재시도합니다. 쓰기 요청은 자동 재시도하지 않습니다.

## 화면

- 상품 탐색: 도메인별 상품, 리뷰·쿼드 수, 긍정 비율, 실제 등록된 리뷰와 쿼드 상세.
- 조건 추천: 카테고리별 특성 선택, AND/OR 조건, 상위 1~50개 상품, 점수와 리뷰 근거.
- 리뷰 분석: 입력 문장을 백엔드에 전달하고 ACOS 쿼드를 표시. 현재 학습 모델 미연결 상태에서는 영어 키워드 중심의 규칙 기준선임을 명시합니다.
- 데이터 가져오기: UTF-8 JSON/TSV 파일을 먼저 미리보기하고 사용자가 가져오기를 눌러야 저장. 파일 최대 5 MiB, 리뷰 1,000개, 리뷰당 쿼드 64개. 정규화된 실제 API 요청의 크기를 백엔드 `/api/meta` 제한(기본 2,000,000 bytes)과 비교하여 초과하면 가져오기를 차단합니다. 한글은 전송 시 ASCII escape로 늘어날 수 있습니다.

도메인이나 연결 설정 변경 시 이전 분석·추천·선택 상품·업로드 상태가 초기화됩니다. 사용자 리뷰는 HTML 문자열에 삽입하지 않으며 결과는 JSON으로 다운로드할 수 있습니다. 예시 상품은 합성 데이터로 표시합니다.

추천 점수는 일치한 쿼드의 `(긍정 수 - 부정 수) / 전체 수`이며 범위는 -1~+1입니다. 구매 확률이나 모델 신뢰도가 아닙니다.

## 가져오기 형식

JSON은 아래 객체 또는 `records` 배열만 사용할 수 있습니다. 각 리뷰에 같은 `product_id`를 사용하면 한 상품으로 묶입니다.

```json
{
  "domain": "laptop",
  "records": [{
    "product_id": "my-laptop-01",
    "product_name": "내 노트북",
    "text": "The battery is great.",
    "quads": [{
      "aspect": "battery",
      "category": "BATTERY#OPERATION_PERFORMANCE",
      "opinion": "great",
      "sentiment": "Positive"
    }]
  }]
}
```

감성은 `Negative`, `Neutral`, `Positive` 또는 `0`, `1`, `2`입니다. 빈 `quads`는 미주석 리뷰로 저장되며 자동 분석하지 않습니다. 암시적인 aspect/opinion은 빈 문자열을 사용합니다. 리뷰는 최대 8,000자, 상품명과 aspect는 200자, opinion은 500자입니다. 상품 ID는 최대 80자로 영문·숫자로 시작하고 영문·숫자·밑줄·마침표·하이픈만 사용합니다. 최종 카테고리·필드 검증과 중복 처리는 백엔드가 담당합니다.

헤더가 있는 TSV는 `product_id`, `product_name`(선택), `text`, `quads` 열을 사용합니다. `quads` 열은 JSON 배열입니다. TSV의 따옴표는 표준 CSV quoting 규칙을 따릅니다.

공식 ACOS TSV도 지원합니다: `문장<TAB>a_start,a_end CATEGORY SENTIMENT o_start,o_end`. 범위는 **끝 제외** `[start, end)`이며 `(-1,-1)`은 암시적 표현입니다. 상품 ID가 없는 문장은 전체 문장의 안정적인 SHA-256 해시로 별도 상품을 생성하므로 실제 상품 단위 추천에는 명시적인 상품 ID가 있는 형식을 권장합니다. `product_id @@@ 문장` 형식에서는 해당 ID를 유지합니다.

## 구조

- `streamlit_app.py`: 화면과 세션 상태.
- `src/api_client.py`: API 설정, 제한 시간과 오류 처리.
- `src/import_data.py`: 업로드 파싱·미리보기용 정규화. 추천 알고리즘은 포함하지 않습니다.
- `assets/style.css`: 정적 화면 스타일.
- 기존 `src/loader.py`, `indexer.py`, `recommender.py`: TSV/집계의 호환용 유틸리티. 현재 화면의 추천 경로는 백엔드만 사용합니다.
