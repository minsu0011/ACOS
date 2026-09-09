# 확인한 범위

기존 Python 3.7 backend 환경에서 Flask test client로 빈 요청 400 및 고정 mock 응답을 직접 확인했습니다. 최신 pytest를 해당 옛 interpreter에 섞으면 문법 오류가 나므로 직접 contract 함수를 실행했습니다. 실제 extraction 모델 정확도는 미검증입니다.

재학습이나 새로운 봉인검증을 이 코드 정리 과정에서 수행하지 않았습니다. 작은 smoke test와 모델 일반화 성능은 별개입니다.

