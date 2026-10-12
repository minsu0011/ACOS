# 실행과 설정

## 한 번에 실행

저장소 루트에서 의존성을 준비하고 run.py를 실행합니다.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe run.py
~~~

API는 5001, 화면은 8501에서 localhost로 열립니다. .env는 run.py가 읽으며 환경변수가 이미 있으면 그 값을 우선합니다. 브라우저를 자동으로 열려면 --open-browser를 추가합니다. 터미널의 Ctrl+C로 두 자식 프로세스를 정리합니다.

## 따로 실행

터미널 1:

~~~powershell
.\.venv\Scripts\python.exe acos-backend/serve.py --port 5001
~~~

터미널 2:

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run acos-frontend/acos_reco/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
~~~

별도 실행에서는 .env를 자동으로 읽지 않으므로 필요한 환경변수를 해당 프로세스에 지정합니다. API가 다른 주소라면 ACOS_API_URL을 설정합니다. 업로드 파일은 사용자 브라우저에서 받은 내용만 검증하며 클라이언트가 서버의 임의 파일 경로를 지정하지 않습니다.

## 환경변수

| 변수 | 기본/의미 |
|---|---|
| ACOS_DATABASE_PATH | data/acos.sqlite3; 서버 내부 저장 경로 |
| ACOS_SEED_DEMO | true; 빈 DB에만 합성 데모 추가 |
| ACOS_ANALYSIS_MODE | heuristic; 규칙 기반 분석 |
| ACOS_PREDICTOR_FACTORY | 신뢰하는 로컬 module:factory. factory()는 callable 또는 predict(sentence,domain)를 가진 객체 반환 |
| ACOS_BACKEND_PORT | 5001 |
| ACOS_FRONTEND_PORT | 8501 |
| ACOS_API_URL | http://127.0.0.1:5001 |
| ACOS_STARTUP_TIMEOUT | 120초; 서비스별 준비 대기 시간, 1~600초 |
| ACOS_ALLOWED_ORIGINS | 제한된 로컬 브라우저 origin; backend 설정 참고 |

키나 사용자 리뷰를 .env.example에 넣지 않습니다. 기본 API는 로그인을 요구하지 않는 로컬 도구입니다. 네트워크 외부에 노출하는 운영 배포는 이 설정만으로 안전하게 완료되지 않습니다.

factory는 model 모드에서만 로드합니다. 클라이언트 요청이 import 경로나 코드를 지정하지 않습니다. 올바른 가중치·환경·어댑터를 직접 준비해야 하며, 로드 실패 시 모델 미연결 상태를 표시하고 분석은 503을 반환합니다. 이 저장소는 predictor 경계만 제공하며 pretrained 연구 모델을 자동 설치하지 않습니다.

통합 실행기는 두 서비스가 준비된 다음 주소를 출력하고 브라우저를 엽니다. 모델 로딩도 백엔드 준비 시간에 포함됩니다. 느린 모델은 `--startup-timeout 300`처럼 대기 시간을 조절할 수 있습니다. 제한 시간을 넘거나 서비스가 종료되면 이 실행기가 만든 두 자식 프로세스만 정리합니다.

## 연구 코드

acos-backend/pipeline.py와 database.py, 두 Extract-Classify 연구 디렉터리는 이전 실험 코드입니다. 모델 파일과 별도 의존성이 필요하고 기존 실험 DB·임시 출력에 영향을 줄 수 있습니다. run.py나 serve.py는 이 코드를 실행하지 않습니다.

앱 동작 확인을 위해 모델을 다운로드하거나 GPU 학습할 필요는 없습니다. 규칙 기반 분석과 실제 모델 성능은 구분해야 합니다.
