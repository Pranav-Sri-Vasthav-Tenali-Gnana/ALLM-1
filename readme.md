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

## Flow Chart (Mermaid) — paste into README

> GitHub renders Mermaid in Markdown. Paste this as-is:

```mermaid
flowchart TD
    A[Raw Dataset] --> B[Preprocessing / Tokenization]
    B --> C{Task Type}

    C -->|Classification| D[Train/Fine-tune Lightweight Model]
    C -->|Summarization| E[Train/Fine-tune Seq2Seq Model]

    D --> F[Evaluate Metrics\nAccuracy/F1]
    E --> G[Evaluate Metrics\nROUGE]

    F --> H[Hyperparameter Tuning\n(lr, batch, epochs, wd)]
    G --> H

    H --> I[Export Model\n(PyTorch -> ONNX optional)]
    I --> J[Quantization\nFP32 -> INT8/uint8]

    J --> K[Benchmarking\nLatency / Throughput / Size]
    K --> L[Model Registry\nVersioned Artifacts + Metadata]

    L --> M[Inference Service\nFastAPI: /infer /benchmark]
    M --> N[UI Dashboard (later)\nInput + Model Choice + Results]

    K --> O[Experiment Report\nTables + Plots]
    O --> P[Paper Draft (later)\nMethod + Results + Discussion]
```