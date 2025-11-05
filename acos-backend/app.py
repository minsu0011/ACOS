from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# predictor.py는 실제 모델 로딩 및 예측 로직을 담당하는 파일입니다. (아래 별도 제공)
# from predictor import ACOS_Predictor 

# --- Flask 앱 설정 ---
app = Flask(__name__)
CORS(app)

# --- 모델 경로 설정 ---
# run.sh 훈련 완료 후 생성된 모델 파일 경로
STEP1_MODEL_DIR = './output/Extract-Classify-QUAD/rest16_1st/' 
STEP2_MODEL_DIR = './output/Extract-Classify-QUAD/rest16_2nd/'

# --- 모델 로딩 (서버 시작 시 1회 실행) ---
predictor = None
# try:
#     if os.path.exists(STEP1_MODEL_DIR) and os.path.exists(STEP2_MODEL_DIR):
#         # predictor = ACOS_Predictor(model_dir_step1=STEP1_MODEL_DIR, model_dir_step2=STEP2_MODEL_DIR)
#         print("실제 모델 로더가 준비되었습니다. (현재 주석 처리됨)")
#     else:
#         print("경고: 훈련된 모델 경로를 찾을 수 없습니다. '/analyze' API는 가상 데이터로 응답합니다.")
# except Exception as e:
#     print(f"모델 로딩 중 오류 발생: {e}")

# --- API 엔드포인트: 실시간 문장 분석 ---
@app.route('/analyze', methods=['POST'])
def analyze_sentence():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'sentence가 누락되었습니다.'}), 400

    sentence = data['sentence']
    
    if predictor:
        # 실제 모델을 사용하여 예측 수행
        try:
            results = predictor.predict(sentence)
            response = {
                'input_sentence': sentence,
                'results': results
            }
            return jsonify(response)
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            return jsonify({'error': '문장 분석 중 오류가 발생했습니다.'}), 500
    else:
        # 모델 로딩 실패 시, 프론트엔드 테스트를 위한 가상 데이터 응답
        print("가상 데이터로 응답합니다.")
        example = {
            "input_sentence": sentence,
            "results": [
                {"aspect": "battery life", "category": "BATTERY#OPERATION_PERFORMANCE", "opinion": "amazing", "sentiment": "Positive"},
                {"aspect": "screen colors", "category": "DISPLAY#QUALITY", "opinion": "dull", "sentiment": "Negative"}
            ]
        }
        return jsonify(example)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)