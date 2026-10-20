"""Inference wrappers -- the single code path used by eval, bench and the API.

``load_predictor(variant_dir)`` reads ``allm_meta.json`` and returns the right
predictor for the task + variant, so callers never branch on fp32/int8/onnx.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.models.factory import auto_model_cls, load_tokenizer, read_meta


def _batches(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class BasePredictor:
    def __init__(self, variant_dir: str | Path, meta: dict[str, Any], model, tokenizer, device: str = "cpu"):
        self.variant_dir = Path(variant_dir)
        self.meta = meta
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def task(self) -> str:
        return self.meta["task"]

    @property
    def variant(self) -> str:
        return self.meta["variant"]

    def _encode(self, texts: list[str], tensors: str = "pt"):
        enc = self.tokenizer(
            [self.meta["prefix"] + t for t in texts],
            truncation=True,
            max_length=self.meta["max_length"],
            padding=True,
            return_tensors=tensors,
        )
        return enc.to(self.device) if tensors == "pt" else enc


class ClassificationPredictor(BasePredictor):
    def logits(self, texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            return self.model(**self._encode(texts)).logits.float().cpu().numpy()

    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        labels = self.meta["label_names"]
        out = []
        for chunk in _batches(texts, batch_size):
            logits = self.logits(chunk)
            probs = _softmax(logits)
            for row in probs:
                idx = int(row.argmax())
                out.append({
                    "label": labels[idx],
                    "label_id": idx,
                    "score": float(row[idx]),
                    "scores": {labels[i]: float(p) for i, p in enumerate(row)},
                })
        return out


class OnnxClassificationPredictor(ClassificationPredictor):
    def logits(self, texts: list[str]) -> np.ndarray:
        enc = self._encode(texts, tensors="np")
        feed = {i.name: enc[i.name].astype(np.int64) for i in self.model.get_inputs()}
        return self.model.run(["logits"], feed)[0]


class SummarizationPredictor(BasePredictor):
    def predict(
        self,
        texts: list[str],
        batch_size: int = 8,
        max_new_tokens: int | None = None,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 0,
    ) -> list[str]:
        max_new_tokens = max_new_tokens or self.meta["max_target_length"]
        out: list[str] = []
        with torch.inference_mode():
            for chunk in _batches(texts, batch_size):
                ids = self.model.generate(
                    **self._encode(chunk),
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    do_sample=False,
                )
                out.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return [s.strip() for s in out]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_predictor(
    variant_dir: str | Path, num_threads: int | None = None, device: str = "cpu"
) -> BasePredictor:
    """Load any variant. ``device`` only applies to fp32 -- int8/onnx kernels are CPU-only."""
    variant_dir = Path(variant_dir)
    meta = read_meta(variant_dir)
    task, variant = meta["task"], meta["variant"]
    if num_threads:
        torch.set_num_threads(num_threads)
    tokenizer = load_tokenizer(variant_dir)

    if variant == "onnx-uint8":
        import onnxruntime as ort

        from core.quant.quantize import ONNX_UINT8

        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
        session = ort.InferenceSession(
            str(variant_dir / ONNX_UINT8), opts, providers=["CPUExecutionProvider"]
        )
        return OnnxClassificationPredictor(variant_dir, meta, session, tokenizer)

    if variant == "int8":
        from core.quant.quantize import load_int8

        model, device = load_int8(variant_dir, task), "cpu"
    else:
        model = auto_model_cls(task).from_pretrained(str(variant_dir)).eval().to(device)

    cls = ClassificationPredictor if task == "classification" else SummarizationPredictor
    return cls(variant_dir, meta, model, tokenizer, device)
