# CPU inference image for ALLM-1. Build context = the bundle made by
# scripts/build_space.py (code + a frozen model registry).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    ALLM_REGISTRY_DIR=/app/registry \
    ALLM_REPORTS_GLOB=reports/*.json \
    ALLM_NUM_THREADS=2

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir "torch==2.7.1" --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user . .
USER user

# Optional: show a GitHub link on the demo page (or set it as a Space variable).
# ENV ALLM_GITHUB_URL=https://github.com/<you>/allm-1

EXPOSE 7860
CMD ["uvicorn", "service.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
