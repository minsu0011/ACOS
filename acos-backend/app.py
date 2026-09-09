from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
predictor = None  # 모델 연결 전에는 UI용 예시만 반환


@app.route('/analyze', methods=['POST'])
def analyze_sentence():
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({'error': 'sentence가 누락되었습니다.'}), 400
    sentence = data['sentence']
    if predictor is not None:
        try:
            return jsonify({'input_sentence': sentence,
                            'results': predictor.predict(sentence)})
        except Exception:
            return jsonify({'error': '문장 분석 중 오류가 발생했습니다.'}), 500
    return jsonify({
        'input_sentence': sentence,
        'results': [
            {'aspect': 'battery life', 'category': 'BATTERY#OPERATION_PERFORMANCE',
             'opinion': 'amazing', 'sentiment': 'Positive'},
            {'aspect': 'screen colors', 'category': 'DISPLAY#QUALITY',
             'opinion': 'dull', 'sentiment': 'Negative'},
        ],
    })


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
