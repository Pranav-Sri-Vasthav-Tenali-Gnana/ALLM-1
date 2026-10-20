"""ALLM-1 inference API.

    uvicorn service.app.main:app --port 8000

Environment:
    ALLM_REGISTRY_DIR         registry root            (default artifacts/registry)
    ALLM_DEFAULT_CLASSIFIER   model ref for /classify  (default agnews-cls)
    ALLM_DEFAULT_SUMMARIZER   model ref for /summarize (default cnndm-sum)
    ALLM_NUM_THREADS          torch/ORT threads        (default 4)
    ALLM_MAX_LOADED           models kept in memory    (default 4)
    ALLM_REPORTS_GLOB         benchmark reports shown in the UI (default artifacts/*/report.json)
    ALLM_GITHUB_URL           repo link shown in the UI (optional)
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from service.app.schemas import (
    MAX_CHARS,
    ClassifyRequest,
    ClassifyResponse,
    ModelInfo,
    SummarizeRequest,
    SummarizeResponse,
)
from service.registry.registry import ModelRegistry, RegistryError, ResolvedModel


class ModelCache:
    """Thread-safe LRU of loaded predictors, keyed by resolved ``name@version``.

    Keying on the *resolved* version means promoting a new version to
    ``production`` takes effect on the next request -- no restart needed.
    """

    def __init__(self, loader: Callable[..., Any], max_loaded: int, num_threads: int):
        self._loader = loader
        self._max = max_loaded
        self._threads = num_threads
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, resolved: ResolvedModel) -> Any:
        with self._lock:
            if resolved.ref in self._items:
                self._items.move_to_end(resolved.ref)
                return self._items[resolved.ref]
            predictor = self._loader(resolved.path, num_threads=self._threads)
            self._items[resolved.ref] = predictor
            if len(self._items) > self._max:
                self._items.popitem(last=False)
            return predictor

    def loaded(self) -> list[str]:
        return list(self._items)


def create_app(registry: ModelRegistry | None = None, loader: Callable[..., Any] | None = None) -> FastAPI:
    """App factory -- tests inject a temp registry and a fake loader."""
    if loader is None:
        from core.models.predictor import load_predictor as loader
    registry = registry or ModelRegistry(os.environ.get("ALLM_REGISTRY_DIR", "artifacts/registry"))
    defaults = {
        "classification": os.environ.get("ALLM_DEFAULT_CLASSIFIER", "agnews-cls"),
        "summarization": os.environ.get("ALLM_DEFAULT_SUMMARIZER", "cnndm-sum"),
    }
    cache = ModelCache(
        loader,
        max_loaded=int(os.environ.get("ALLM_MAX_LOADED", 4)),
        num_threads=int(os.environ.get("ALLM_NUM_THREADS", 4)),
    )

    reports_glob = os.environ.get("ALLM_REPORTS_GLOB", "artifacts/*/report.json")
    github_url = os.environ.get("ALLM_GITHUB_URL", "")

    app = FastAPI(title="ALLM-1 Inference API", version="1.0.0")

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
        return response

    def resolve(ref: str | None, task: str) -> ResolvedModel:
        try:
            resolved = registry.resolve(ref or defaults[task])
        except RegistryError as e:
            raise HTTPException(status_code=404, detail=str(e.args[0])) from None
        if resolved.task != task:
            raise HTTPException(400, f"{resolved.ref} is a {resolved.task} model, not {task}")
        return resolved

    def check_lengths(texts: list[str]) -> None:
        if any(not t.strip() for t in texts):
            raise HTTPException(422, "texts must be non-empty")
        if any(len(t) > MAX_CHARS for t in texts):
            raise HTTPException(413, f"each text must be <= {MAX_CHARS} characters")

    def info(r: ResolvedModel) -> ModelInfo:
        return ModelInfo(name=r.name, version=r.version, variant=r.variant, task=r.task)

    @app.get("/", include_in_schema=False)
    def demo_page() -> FileResponse:
        # no-cache: browsers revalidate, so a redeploy shows up immediately.
        return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "loaded_models": cache.loaded()}

    def load_reports() -> list[dict[str, Any]]:
        reports = []
        for path in sorted(Path().glob(reports_glob)):
            report = json.loads(path.read_text(encoding="utf-8"))
            if not report.get("name", "").startswith("smoke"):
                reports.append(report)
        return reports

    @app.get("/v1/info")
    def project_info() -> dict[str, Any]:
        """Everything the demo page needs: served models, benchmark reports, links."""
        served = {}
        for task, ref in defaults.items():
            try:
                r = registry.resolve(ref)
            except RegistryError:
                continue
            version = registry.list_models()[r.name]["versions"][str(r.version)]
            served[task] = {**info(r).model_dump(), "metrics": version.get("metrics", {}),
                            "base_model": version.get("base_model")}
        return {"served": served, "reports": load_reports(), "github_url": github_url}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"defaults": defaults, "models": registry.list_models()}

    # Sync handlers: FastAPI runs them in a threadpool, so CPU-bound inference
    # doesn't block the event loop.
    @app.post("/v1/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        check_lengths(req.texts)
        resolved = resolve(req.model, "classification")
        predictor = cache.get(resolved)
        start = time.perf_counter()
        preds = predictor.predict(req.texts)
        latency = (time.perf_counter() - start) * 1000
        if not req.return_all_scores:
            preds = [{**p, "scores": None} for p in preds]
        return ClassifyResponse(model=info(resolved), predictions=preds, latency_ms=round(latency, 2))

    @app.post("/v1/summarize", response_model=SummarizeResponse)
    def summarize(req: SummarizeRequest) -> SummarizeResponse:
        check_lengths(req.texts)
        resolved = resolve(req.model, "summarization")
        predictor = cache.get(resolved)
        start = time.perf_counter()
        summaries = predictor.predict(
            req.texts,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
            no_repeat_ngram_size=req.no_repeat_ngram_size,
        )
        latency = (time.perf_counter() - start) * 1000
        return SummarizeResponse(model=info(resolved), summaries=summaries, latency_ms=round(latency, 2))

    return app


app = create_app()
