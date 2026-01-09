### ALLM-1 — Adaptive Lite Language Model (v1)

**ALLM-1** is a lightweight text understanding system focused on **classification** and **summarization**, designed to be **small, fast, and practical to deploy**.

The goal is to build an end-to-end pipeline that covers:

* **Fine-tuning** small transformer models on domain/task datasets
* **Hyperparameter tuning** to maximize accuracy within constraints
* **INT8/uint8 quantization** to reduce model size and improve inference latency
* **System design for serving**: versioned models, reproducible benchmarking, and a clean inference API
* A minimal **full-stack dashboard** to run inference and view results (later phase)
* A **paper-ready evaluation** comparing FP32 vs INT8 across accuracy/ROUGE, latency, and model size

This repo is intentionally modular so future extensions (RLHF, RLAIF, DPO) can plug into the same training + evaluation + deployment framework.

---

## Key Outcomes (what this project will prove)

* **Accuracy vs efficiency trade-offs** (FP32 vs INT8)
* **Latency benchmarks** on CPU inference
* **Model size reduction** with quantization
* Clean reproducibility: scripts + configs + saved artifacts

---

## Planned Modules

* `core/` → training, evaluation, tuning, quantization, benchmarking
* `service/` → FastAPI inference server + model registry
* `ui/` → minimal dashboard (later)
* `paper/` → paper draft + experiment tables (later)

---

## Repo structure (lock this early)

```
allm-1/
  core/
    data/                 # dataset adapters + preprocessing
    models/               # model wrappers / configs
    train/                # training entrypoints
    eval/                 # evaluation + metrics
    tune/                 # hyperparameter search
    quant/                # quantization pipelines
    bench/                # latency/throughput/size benchmarks
    utils/                # logging, seeding, I/O helpers
  service/
    app/                  # FastAPI app
    registry/             # model registry interface
  ui/                     # later (optional)
  paper/
    draft.md
    tables/
    figures/
  configs/
    classification/
    summarization/
  scripts/
  artifacts/              # gitignored (local) or managed via DVC later
  tests/
  README.md
  Makefile (optional)
```