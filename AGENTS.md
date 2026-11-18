# AGENTS.md

Instructions for AI coding agents working on ML-Gateway, whatever the tool.

## Rules for AI agents

These rules come from the "Spirit of the project" and "Use of AI and LLM tools"
sections of [CONTRIBUTING.md](CONTRIBUTING.md). They are mandatory. Read CONTRIBUTING.md
too: it is the reference, and it wins if the two files ever disagree.

- **Keep the spirit of the project.** The gateway serves models; it does not train
  them. Every model is loaded at startup, and every endpoint that runs a model goes
  through the inference gate (`api/throttle.py`): never add a path that loads a model
  during a request or runs one outside the gate. The bi-encoder scoring contract comes
  from VulnTrain and is reproduced exactly: do not adjust it here. The tests stub the
  model layer and download nothing: keep them that way. Add no dependency without a
  justification.
- **Do not weaken the protections.** Never relax the token check of the index endpoint
  (`api/security.py`), the admission control, or the allow-lists of loadable models.
  Never write a token or a host name of a deployment to a tracked file.
- **Work from an issue.** A change should follow an issue that explains its rationale
  (the problem, why it matters, the proposed approach); only small fixes can skip it.
- **Stay in scope.** Keep changes minimal and focused on the task. Do not reformat,
  rename, or rewrite unrelated code or comments.
- **Check what you produce.** Run `poetry run mypy api/ tests/` and `poetry run pytest`
  before handing your work over, and say plainly what you could not verify: a model
  you did not load, a timing you did not measure, a behaviour under load you did not
  observe. The human you work with must read, understand and test every change before
  submitting it: vibe-coded pull requests are closed.
- **Make it explainable.** Explain every change to the human you work with. Reviewers
  may ask them to explain any part of it, and a pull request whose author cannot is
  closed. Do not write their answers to review comments for them to paste.
- **A human reviews and submits.** Never open a pull request or an issue on your own,
  and never make a release; the decision is the human's, after they have reviewed your
  work.
- **Report only what was reproduced.** When you help write an issue, the human must
  have reproduced the problem. Never present an unverified finding, or a figure that
  was not measured, as a confirmed one.
- **Say so.** End every commit you author with a `Co-Authored-By` trailer naming the
  model and its version, for example:

  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```

  When you draft a pull request description, follow
  [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) and keep all its
  sections, and mention that a significant part of the change was generated with an AI
  tool. Never tick the box "I have read, understood and tested every change": it is
  the human's own statement, leave it for them.

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

# Download one revision (serve it by pinning the same SHA in ML_GATEWAY_MODEL_REVISIONS)
poetry run ml-gw-cli refresh-model --model-name "CIRCL/vulnerability-severity-classification-RoBERTa-base" --revision <sha>

# Fill the bi-encoder index from Vulnerability-Lookup NDJSON dumps / an .npz of vectors
poetry run ml-gw-cli backfill-index --dumps /path/to/dumps/
poetry run ml-gw-cli embed-dumps --dumps /path/to/dumps/ --output vectors.npz --device cuda  # on a GPU host
poetry run ml-gw-cli import-index --file vectors.npz

# Time single forward passes on this host (thread count and oneDNN selectable)
poetry run ml-gw-cli bench --threads 4

# Check that int8 quantization runs on this CPU before setting ML_GATEWAY_QUANTIZE=1
poetry run ml-gw-cli check-quantization

# Type checking
poetry run mypy api/ tests/

# Run the test suite (stubs the model layer; no model downloads)
poetry run pytest

# Docker deployment
docker compose up -d
```

## Architecture

The `api/` package follows a layered architecture:

- **Routers** (`routers/classification_router.py`, `routers/retrieval_router.py`, `routers/stats_router.py`) — FastAPI endpoint definitions. Routes to service layer; `GET /stats` exposes the per-worker cache `cache_info()` counters and the gate's counters.
- **Services** (`services/classification_service.py`, `services/retrieval_service.py`) — Business logic. Selects model, formats output. The retrieval service also owns the per-model vector store (opened lazily under `ML_GATEWAY_INDEX_DIR`, default `./index`).
- **Models** (`models/severity_model.py`, `models/attack_model.py`, `models/biencoder_model.py`) — `SeverityClassifier`, `AttackTechniqueClassifier` and `AttackBiEncoder` wrap Hugging Face transformers. Models are lazy-loaded and cached in per-module in-memory `_model_cache` dicts keyed by model name. The `LABELS` dict is the severity model registry mapping model names to their label sets; the attack classifier reads labels from the model config's `id2label` and gates loadable repos through the `ATTACK_MODELS` allow-list; the bi-encoder reads its scoring constants from the config's `biencoder` block and is gated by `BIENCODER_MODELS`. `api/data/attack_technique_names.json` maps technique IDs to official ATT&CK names; `api/data/attack_technique_texts.json` holds the `"Name. Description"` text per active enterprise technique (built from the MITRE STIX bundle, markup stripped) used to embed techniques outside the bi-encoder's trained vocabulary.
- **Revisions** (`models/revisions.py`) — `pinned_revision(model_name)` reads `ML_GATEWAY_MODEL_REVISIONS` (`<model>=<sha>` pairs); every wrapper and the CLI's `_refresh` pass it as `revision=` to `from_pretrained`, so the cache and the server agree on the snapshot.
- **Quantization** (`models/quantization.py`) — `ML_GATEWAY_QUANTIZE=1` converts the two classifiers to dynamic int8 via `SeverityClassifier.quantize()` / `AttackTechniqueClassifier.quantize()`, called from the lifespan in each worker after the fork (never in the preloading master); the bi-encoder stays fp32.
- **Throttle** (`throttle.py`) — `InferenceGate`: per-worker admission control for every endpoint that runs a model (`ML_GATEWAY_INFERENCE_CONCURRENCY` running; a call is refused with `503` + `Retry-After` when the queued work, at the worker's measured service time, exceeds `ML_GATEWAY_INFERENCE_MAX_WAIT` seconds, or `ML_GATEWAY_INFERENCE_QUEUE` calls wait). Handlers are `async def` and hand the synchronous service call to `INFERENCE_GATE.run`; endpoints that run no model (root, technique list) bypass it.
- **Availability** (`availability.py`) — `require_enabled`, a router-level dependency: routes listed in `ML_GATEWAY_DISABLED_ENDPOINTS` (route templates, `{id}` placeholders normalised) answer `503` before the gate.
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
