# API 계약

기본 주소는 http://127.0.0.1:5001 입니다. POST는 application/json 객체만 받습니다. 애플리케이션의 실패 응답은 error와 code를 포함한 JSON이며 내부 traceback을 반환하지 않습니다. Waitress가 요청을 먼저 거절하는 크기 초과 오류는 일반 텍스트 413 또는 업로드 중 연결 종료일 수 있습니다. 화면에서는 실제 전송 크기를 먼저 확인합니다.

| 요청 | 용도 |
|---|---|
| GET /api/health | DB 접근과 분석 모드·모델 연결 상태 |
| GET /api/meta | 도메인, 합성 데모 안내, 입력 한도 |
| GET /api/products?domain=laptop | 상품·리뷰 수·쿼드 수·긍정 비율 |
| GET /api/products/{product_id} | 상품 정보·리뷰·쿼드·특성 집계 |
| GET /api/categories?domain=laptop | 카테고리별 특성과 감정 집계 |
| POST /api/analyze | 입력 문장의 규칙/외부 predictor 분석 |
| POST /analyze | 이전 클라이언트와 호환되는 분석 경로 |
| POST /api/recommend | 카테고리·특성 선택에 따른 근거 기반 추천 |
| POST /api/import | 검증된 리뷰·쿼드 배치 저장 |

도메인은 laptop 또는 rest16입니다. 상품 ID와 입력 문자열, 카테고리·감정, 추가 필드와 요청 크기를 검증합니다. 클라이언트는 서버의 파일 경로를 지정할 수 없습니다.

## 분석

~~~json
{"sentence":"The battery is amazing but the screen is dull.","domain":"laptop"}
~~~

응답은 input_sentence, domain, results, analysis_mode, notice를 포함합니다. results의 항목은 aspect, category, opinion, sentiment입니다. 감정은 Positive, Neutral, Negative 중 하나입니다. 알 수 없는 문장은 빈 results가 될 수 있습니다.

## 추천

~~~json
{"domain":"laptop","selections":{"BATTERY#OPERATION_PERFORMANCE":"battery life"},"mode":"AND","top_k":10}
~~~

실제 선택 가능한 feature는 /api/categories에서 조회합니다. AND는 선택 카테고리 전체가 일치해야 하며 OR은 일부 일치를 허용합니다. top_k는 1~50, 선택은 1~16개입니다.

items에 상품 정보, score, matched_categories, evidence, evidence_count가 있습니다. 점수는 일치 쿼드의 (긍정-부정)/전체 평균으로 -1~1 범위이며, 긍정 비율과도 다릅니다. 모든 감정의 일치 근거를 포함하고 score가 큰 순서로 정렬합니다. evidence는 대표 최대 12개이며 전체 수는 evidence_count입니다.

## 가져오기

~~~json
{
  "domain":"laptop",
  "records":[{
    "product_id":"sample-laptop",
    "product_name":"직접 준비한 예시",
    "text":"The battery is great.",
    "quads":[{"aspect":"battery","category":"BATTERY#OPERATION_PERFORMANCE","opinion":"great","sentiment":"Positive"}]
  }]
}
~~~

전체 배치를 검증한 뒤 transaction으로 반영합니다. 동일 상품/리뷰/쿼드는 중복 삽입하지 않습니다. 이미 있는 product_id의 이름·도메인을 다른 값으로 바꾸려 하면 409를 반환하고 전체 요청을 되돌립니다. imported에 products, reviews, quads, records, skipped_records가 있습니다.

요청 본문은 2,000,000 bytes 이하, 문장은 최대8,000자, import는 최대1,000리뷰, 리뷰당 최대64쿼드입니다. 상세 제한은 /api/meta의 limits로 조회합니다. 큰 데이터는 검증 후 명시적으로 나누어 가져오세요. 원 자료에 잘못된 span·감정이 있으면 먼저 확인·수정해야 하며, 모든 기존 benchmark가 무조건 통과한다고 보장하지 않습니다.

## 모델 경계

기본 heuristic은 규칙 분석입니다. model 모드에서 실제 predictor가 없으면 상태는 degraded이고 분석은503을 반환합니다. 연결된 predictor의 실패·잘못된 출력은500입니다. 실제 모델이 실패했다고 조용히 규칙 결과로 바꾸지 않습니다.
