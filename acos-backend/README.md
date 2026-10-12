# Backend

현재 앱은 app.py의 Flask 팩토리와 acos_app 패키지를 사용합니다. serve.py는 로컬 Waitress 실행 파일입니다. 설치·통합 실행 방법은 저장소 루트 README에 있습니다.

pipeline.py, database.py, Data/sample_jsonl.py는 이전 연구·실험 코드입니다. 별도 모델·데이터가 필요하며 기본 애플리케이션이 import하거나 실행하지 않습니다. 특히 이전 실험의 DB/임시 파일 생성·삭제 동작을 현재 서비스의 저장 동작으로 혼동하지 마세요.

ACOS-main의 upstream 코드와 실제 모델 가중치는 규칙 기반 앱의 동작 확인과 분리합니다.
