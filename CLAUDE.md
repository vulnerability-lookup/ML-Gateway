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

# Type checking
poetry run mypy api/

# Docker deployment
docker compose up -d
```

## Architecture

The `api/` package follows a layered architecture:

- **Routers** (`routers/classification_router.py`) — FastAPI endpoint definitions. Routes to service layer.
- **Services** (`services/classification_service.py`) — Business logic. Selects model, formats output.
- **Models** (`models/severity_model.py`) — `SeverityClassifier` wraps Hugging Face transformers. Models are lazy-loaded and cached in an in-memory `_model_cache` dict keyed by model name. The `LABELS` dict is the model registry mapping model names to their label sets.
- **Schemas** (`schemas.py`) — Pydantic request/response models. Default model is `CIRCL/vulnerability-severity-classification-RoBERTa-base`.

Entry points: `api.main:app` (FastAPI), `api.cli:app` (Typer CLI registered as `ml-gw-cli`).

## Supported Models

| Model | Language | Labels |
|---|---|---|
| `CIRCL/vulnerability-severity-classification-RoBERTa-base` | English | Low, Medium, High, Critical |
| `CIRCL/vulnerability-severity-classification-chinese-macbert-base` | Chinese | 低, 中, 高 |
| `CIRCL/vulnerability-severity-classification-russian-ruRoberta-large` | Russian | Low, Medium, High, Critical |

Adding a model requires entries in both `LABELS` in `severity_model.py` and the `models` list in `cli.py:refresh_all`.

## Code Standards

- **mypy strict mode** is enabled (see `[tool.mypy]` in `pyproject.toml`). All code must pass strict type checking.
- Max line length: 120 characters.
- Commit messages follow the pattern: `prefix: [scope] description` (e.g., `chg: [dependencies] Updated dependencies.`, `new: [api] Added endpoint.`, `fix: [model] Fixed label mapping.`).
