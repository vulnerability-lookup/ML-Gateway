# Changelog


## Release 1.0.0 (2026-04-17)

First stable release. The focus of this release is production deployment:
serving many concurrent clients on a multi-core host without paying the
full model-loading cost once per worker, and without re-running inference
for inputs the server has already seen.

### Production deployment with gunicorn + --preload

- Models are now eagerly loaded at module import time via a new
  `preload_models()` function in `api/models/severity_model.py`, called
  from `api/main.py` before the FastAPI app is constructed.
- Combined with gunicorn's `--preload` flag, every model is loaded once
  in the master process before workers are forked. Forked workers
  inherit the already-loaded tensors via copy-on-write, so the models
  occupy memory roughly once instead of once per worker. On a 16-core
  host running 4 workers this is a 4× reduction in model memory.
- Without `--preload`, the new eager loading means each worker still
  loads every model at startup rather than lazily on first request.
  This trades a slower boot for predictable first-request latency and
  removes a class of "first request after restart is slow" issues.
- The README now recommends a production command tuned for 16 cores:
  4 uvicorn workers with `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`
  (4 workers × 4 PyTorch intra-op threads = 16 threads, keeping every
  core busy during inference while avoiding the memory overhead of
  8+ worker processes), plus `--preload`, `--reuse-port` and
  `--proxy-protocol`.

### Per-worker LRU cache for inference results

- Added an in-process `functools.lru_cache` (bounded at 10,000 entries)
  in `api/services/classification_service.py`, keyed by
  `(model_name, description)`. Repeat requests for the same
  description against the same model now return the cached result
  without re-running the model.
- Tradeoff: the cache is per worker, not shared across processes. With
  gunicorn's `--reuse-port` the kernel scatters incoming connections
  across workers, so duplicate requests only benefit from the cache
  when they happen to land on the same worker. In practice this is
  still a win because hot descriptions (e.g. re-enrichment passes over
  the same CVE) repeat often enough to land on the same worker
  multiple times, but users who need cross-worker deduplication should
  front ML-Gateway with an external cache such as Redis.
- Memory footprint is bounded: 10,000 entries × ~1–2 KB per
  description ≈ 10–20 MB of cache per worker. The cache can be
  inspected or cleared at runtime with
  `_cached_predict.cache_info()` and `_cached_predict.cache_clear()`.
- Exceptions (e.g. unknown model name) are not cached, so error
  handling behaviour is unchanged.

### Model registry cleanup

- Removed `CIRCL/vulnerability-severity-classification-distilbert-base-uncased`
  from the `LABELS` registry. The corresponding Hugging Face repository
  is no longer published, so any attempt to load the model (including
  the new eager preload) failed with a 404. The entry was already
  commented out of the CLI's `refresh-all` command and was effectively
  unreachable.


## Release 0.5.0 (2026-04-06)

Added support for the Russian vulnerability severity classification model
(CIRCL/vulnerability-severity-classification-russian-ruRoberta-large).

- Registered the Russian ruRoBERTa-large model in the model registry with standard CVSS severity labels (Low, Medium, High, Critical).
- Added the model to the CLI refresh-all command for pre-downloading.


## Release 0.4.0 (2025-06-27)

Multilingual model support for severity classification.

- Added support for specifying a Hugging Face model via a new model field in the SeverityRequest payload.
- Default model remains CIRCL/vulnerability-severity-classification-RoBERTa-base.
- New models (e.g., Chinese-language severity classifiers) can now be selected dynamically at request time.
- Introduced a get_model_instance() registry to load and cache models on demand.
- Preserved clean architecture with model selection handled in the service layer.


## Release 0.3.0 (2025-05-26)

Added a cli with two commands in order to refresh the models from Hugging Face.
A command to force-refresh a specific model and a command to force-refresh all preconfigured models.


## Release 0.2.0 (2025-05-23)

Refactored ``severity_model.py`` to use a manual inference in order
to have full control over performance, consistency, and debugging.
pytorch is directly used instead of the transformers Pipeline. The Pipeline
was using internal heuristics to map logits to labels (via softmax + sorting)
and other approximations (rounding, label normalization, etc.).


## Release 0.1.0 (2025-05-22)

First working release release.
It provides a service and router for a text classification model.
