# 로컬 데이터

새 애플리케이션은 상품·리뷰·ACOS 결과를 SQLite에 저장합니다. 기본 위치는 data/acos.sqlite3이며 Git에서 제외합니다. 빈 DB에만 직접 작성한 합성 데모를 넣어 실행 흐름을 확인합니다. 기존 DB를 삭제하거나 다시 초기화하지 않습니다.

JSON/TSV 가져오기는 화면에서 미리보기 후 실행합니다. 공식 ACOS TSV의 span은 [start,end)이고 감정은 0=negative, 1=neutral, 2=positive입니다. 상품 ID가 없는 리뷰 문장은 안정된 digest로 구분하지만 실제 상품 정보가 생긴 것은 아닙니다.

학습 TSV, pretrained weights, 사용자 원본 리뷰와 개인 DB는 현재 배포 파일에 포함하지 않습니다. 이전 공개 커밋에는 upstream benchmark TSV가 존재했던 이력이 있으므로 Git 이력까지 데이터가 전혀 없다고 주장하지 않습니다.

NUSTM ACOS Restaurant/Laptop schema를 참고하며 원천의 이용 조건을 따릅니다. 기본 규칙 분석과 연구 모델의 실제 추론·성능은 별개입니다.
