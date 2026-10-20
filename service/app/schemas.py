"""Request / response models for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, Field

MAX_BATCH = 64
MAX_CHARS = 20_000


class ModelInfo(BaseModel):
    name: str
    version: int
    variant: str
    task: str


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    model: str | None = Field(None, description="name, name@version or name@alias")
    return_all_scores: bool = False


class ClassPrediction(BaseModel):
    label: str
    score: float
    scores: dict[str, float] | None = None


class ClassifyResponse(BaseModel):
    model: ModelInfo
    predictions: list[ClassPrediction]
    latency_ms: float


class SummarizeRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=16)
    model: str | None = None
    max_new_tokens: int = Field(64, ge=8, le=256)
    num_beams: int = Field(1, ge=1, le=4)
    # Blocks repeated n-grams, the usual fix for greedy decoding loops.
    # Evaluation runs without it, so reported ROUGE reflects the raw model.
    no_repeat_ngram_size: int = Field(3, ge=0, le=10)


class SummarizeResponse(BaseModel):
    model: ModelInfo
    summaries: list[str]
    latency_ms: float
