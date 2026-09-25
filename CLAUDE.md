# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML-Gateway is a FastAPI server that loads pre-trained Hugging Face NLP models and exposes them via a REST API for vulnerability severity classification. It serves the [Vulnerability-Lookup](https://github.com/vulnerability-lookup) project.

## Development Commands

```bash
# Install dependencies (requires Poetry 2.0+)
poetry install

# Run the server
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000

# Pre-download/refresh all models
poetry run ml-gw-cli refresh-all

# Refresh a specific model
poetry run ml-gw-cli refresh-model --model-name "CIRCL/vulnerability-severity-classification-RoBERTa-base"

# Fill the bi-encoder index from Vulnerability-Lookup NDJSON dumps / an .npz of vectors
poetry run ml-gw-cli backfill-index --dumps /path/to/dumps/
poetry run ml-gw-cli embed-dumps --dumps /path/to/dumps/ --output vectors.npz --device cuda  # on a GPU host
poetry run ml-gw-cli import-index --file vectors.npz

# Type checking
poetry run mypy api/ tests/

# Run the test suite (stubs the model layer; no model downloads)
poetry run pytest

# Docker deployment
docker compose up -d
```

## Architecture

The `api/` package follows a layered architecture:

- **Routers** (`routers/classification_router.py`, `routers/retrieval_router.py`) — FastAPI endpoint definitions. Routes to service layer.
- **Services** (`services/classification_service.py`, `services/retrieval_service.py`) — Business logic. Selects model, formats output. The retrieval service also owns the per-model vector store (opened lazily under `ML_GATEWAY_INDEX_DIR`, default `./index`).
- **Models** (`models/severity_model.py`, `models/attack_model.py`, `models/biencoder_model.py`) — `SeverityClassifier`, `AttackTechniqueClassifier` and `AttackBiEncoder` wrap Hugging Face transformers. Models are lazy-loaded and cached in per-module in-memory `_model_cache` dicts keyed by model name. The `LABELS` dict is the severity model registry mapping model names to their label sets; the attack classifier reads labels from the model config's `id2label` and gates loadable repos through the `ATTACK_MODELS` allow-list; the bi-encoder reads its scoring constants from the config's `biencoder` block and is gated by `BIENCODER_MODELS`. `api/data/attack_technique_names.json` maps technique IDs to official ATT&CK names; `api/data/attack_technique_texts.json` holds the `"Name. Description"` text per active enterprise technique (built from the MITRE STIX bundle, markup stripped) used to embed techniques outside the bi-encoder's trained vocabulary.
- **Throttle** (`throttle.py`) — `InferenceGate`: per-worker admission control for every endpoint that runs a model (`ML_GATEWAY_INFERENCE_CONCURRENCY` running, `ML_GATEWAY_INFERENCE_QUEUE` waiting, then `503` + `Retry-After`). Handlers are `async def` and hand the synchronous service call to `INFERENCE_GATE.run`; endpoints that run no model (root, technique list) bypass it.
- **Store** (`store/vector_store.py`) — `VectorStore`: append-only float16 matrix + ID list on disk, memory-mapped and shared by all worker processes, brute-force cosine search, advisory file lock for writers, pinned to one model revision via `meta.json`.
- **Backfill** (`backfill.py`) — dump record extraction per feed layout, the batch backfill loop, the `.npz` writer and import used by the `backfill-index` / `embed-dumps` / `import-index` CLI commands.
- **Schemas** (`schemas.py`) — Pydantic request/response models. Default model is `CIRCL/vulnerability-severity-classification-RoBERTa-base`.

The bi-encoder scoring contract (mean pooling, per-side truncation lengths, `sigmoid(logit_scale · cos + logit_bias)`) comes from VulnTrain's `docs/attack-biencoder-retrieval.md` and must be reproduced exactly.

Entry points: `api.main:app` (FastAPI), `api.cli:app` (Typer CLI registered as `ml-gw-cli`).

## Supported Models

| Model | Language | Labels |
|---|---|---|
| `CIRCL/vulnerability-severity-classification-RoBERTa-base` | English | Low, Medium, High, Critical |
| `CIRCL/vulnerability-severity-classification-chinese-macbert-base` | Chinese | 低, 中, 高 |
| `CIRCL/vulnerability-severity-classification-russian-ruRoberta-large` | Russian | Low, Medium, High, Critical |
| `CIRCL/vulnerability-attack-technique-classification-roberta-base` | English | ATT&CK technique IDs (multi-label, from `id2label`) |
| `CIRCL/vulnerability-attack-technique-biencoder` | English | none — embeddings for retrieval (53 trained techniques) |

Adding a model means adding it to the matching registry (`LABELS`, `ATTACK_MODELS` or `BIENCODER_MODELS`); `ml-gw-cli refresh-all` iterates over those registries.

## Code Standards

- **mypy strict mode** is enabled (see `[tool.mypy]` in `pyproject.toml`). All code must pass strict type checking.
- Max line length: 120 characters.
- Commit messages follow the pattern: `prefix: [scope] description` (e.g., `chg: [dependencies] Updated dependencies.`, `new: [api] Added endpoint.`, `fix: [model] Fixed label mapping.`).
