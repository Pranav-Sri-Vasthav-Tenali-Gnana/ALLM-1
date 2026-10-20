---
title: ALLM-1
emoji: ⚡
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: News classification + summarization, quantized for CPU
---

# ALLM-1: Adaptive Lite Language Model

Two small models, fine-tuned and benchmarked FP32 vs INT8 on CPU.

**Classification** (AG News: World / Sports / Business / Sci/Tech), DistilBERT

| Variant | Accuracy | Size | p50 latency (1 input) |
|---|---|---|---|
| FP32 | 91.7% | 255 MB | 54 ms |
| INT8 (served) | 91.6% | 132 MB | 34 ms |

**Summarization** (CNN/DailyMail), T5-small

| Variant | ROUGE-1 / 2 / L | Size | p50 latency (1 article) |
|---|---|---|---|
| FP32 (served) | 40.4 / 19.5 / 29.4 | 231 MB | 1.2 s |
| INT8 | 34.3 / 14.3 / 24.6 | 121 MB | 2.5 s |

INT8 is nearly free for classification but hurts T5 (activation outliers; per-token
decoding overhead), so the registry serves FP32 for summarization.

* Demo page: `/`
* Interactive API docs: `/docs`
* `POST /v1/classify` with `{"texts": ["..."], "return_all_scores": true}`
* `POST /v1/summarize` with `{"texts": ["..."], "max_new_tokens": 64}`
