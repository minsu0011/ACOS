import importlib.util
from pathlib import Path

def test_mock_contract_not_model_inference():
    path = Path(__file__).resolve().parents[1] / 'acos-backend' / 'app.py'
    spec = importlib.util.spec_from_file_location('portfolio_acos_app', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.predictor is None
    client = module.app.test_client()
    assert client.post('/analyze', json={}).status_code == 400
    reply = client.post('/analyze', json={'sentence': 'synthetic contract check'})
    assert reply.status_code == 200
    assert reply.json['input_sentence'] == 'synthetic contract check'
    assert len(reply.json['results']) == 2
