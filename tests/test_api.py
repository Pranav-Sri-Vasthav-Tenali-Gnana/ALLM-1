import pytest
from fastapi.testclient import TestClient

from service.app.main import create_app
from service.registry.registry import ModelRegistry


class FakeClassifier:
    def predict(self, texts, batch_size=32):
        return [{"label": "pos", "label_id": 1, "score": 0.9, "scores": {"neg": 0.1, "pos": 0.9}} for _ in texts]


class FakeSummarizer:
    def predict(self, texts, max_new_tokens=64, num_beams=1, no_repeat_ngram_size=0):
        return [t[:10] for t in texts]


@pytest.fixture
def client(tmp_path, variant_factory):
    reg = ModelRegistry(tmp_path / "registry")
    reg.register("agnews-cls", variant_factory("classification"), alias="production")
    reg.register("cnndm-sum", variant_factory("summarization"), alias="production")
    loads = []

    def loader(path, num_threads=None):
        loads.append(path)
        return FakeSummarizer() if "cnndm-sum" in str(path) else FakeClassifier()

    c = TestClient(create_app(registry=reg, loader=loader))
    c.loads = loads
    return c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_classify(client):
    r = client.post("/v1/classify", json={"texts": ["great game", "stocks fell"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == {"name": "agnews-cls", "version": 1, "variant": "int8", "task": "classification"}
    assert [p["label"] for p in body["predictions"]] == ["pos", "pos"]
    assert body["predictions"][0]["scores"] is None
    assert "X-Process-Time-Ms" in r.headers


def test_model_is_cached(client):
    client.post("/v1/classify", json={"texts": ["a"]})
    client.post("/v1/classify", json={"texts": ["b"]})
    assert len(client.loads) == 1


def test_summarize(client):
    r = client.post("/v1/summarize", json={"texts": ["A long article about things."]})
    assert r.status_code == 200
    assert r.json()["summaries"] == ["A long art"]


def test_errors(client):
    assert client.post("/v1/classify", json={"texts": []}).status_code == 422
    assert client.post("/v1/classify", json={"texts": ["  "]}).status_code == 422
    assert client.post("/v1/classify", json={"texts": ["x"], "model": "nope"}).status_code == 404
    # Task mismatch: a summarizer requested on the classify endpoint.
    assert client.post("/v1/classify", json={"texts": ["x"], "model": "cnndm-sum"}).status_code == 400


def test_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>ALLM-1</title>" in r.text


def test_info(client):
    body = client.get("/v1/info").json()
    assert body["served"]["classification"]["name"] == "agnews-cls"
    assert body["served"]["summarization"]["variant"] == "int8"
    assert isinstance(body["reports"], list)
