# Changelog


## Release 1.3.0 (2026-07-17)

New endpoint: MITRE ATT&CK technique classification.

- `POST /classify/attack-techniques` ranks ATT&CK (Enterprise) techniques
  for a vulnerability description using
  `CIRCL/vulnerability-attack-technique-classification-roberta-base`
  (multi-label; sigmoid scores, `top_k` selectable, 0.5 prediction
  threshold flagged per technique). Each technique is returned with its
  ID, official name, score, and `predicted` flag, alongside the same
  `model` / `model_revision` provenance fields as `/classify/severity`.
- Technique names are resolved from a bundled `id -> name` table
  extracted from the MITRE enterprise ATT&CK STIX data
  (`api/data/attack_technique_names.json`).
- The model is preloaded at startup (gunicorn `--preload` compatible)
  and included in `ml-gw-cli refresh-all`.


## Release 1.2.0 (2026-05-20)

Classification responses now carry the provenance of the model that
produced them, so callers can pin and audit which exact weights generated
a given result.

- `POST /classify/severity` responses now include two new fields:
  - `model`: the Hugging Face repository identifier that produced the
    prediction (e.g.
    `CIRCL/vulnerability-severity-classification-RoBERTa-base`).
  - `model_revision`: the commit SHA of the snapshot that was loaded
    from the Hugging Face Hub at startup, or `null` when the source
    does not carry revision metadata (for example, models loaded from
    a local path).
- The revision is captured once in `SeverityClassifier.__init__` from
  `model.config._commit_hash`, which transformers stamps onto the
  config during `from_pretrained` for both fresh downloads and cached
  snapshots. No extra Hugging Face API call is made at request time.
- A new `SeverityResponse` Pydantic schema is wired as the endpoint's
  `response_model`, so the OpenAPI documentation now describes every
  response field, including the provenance metadata and the optional
  `error` field returned when an unknown model is requested.
- `error` responses (unknown model) now also include `model` and
  `model_revision` (the latter as `null`) so the response shape stays
  consistent across success and failure paths.

This is a backwards-compatible change for clients that read only
`severity` and `confidence`; the new keys are additive.


## Release 1.1.0 (2026-05-12)

Adds a time-to-live to the per-worker inference cache to bound how long
stale entries can sit in memory between requests, in addition to the
existing size cap.

- Replaced `functools.lru_cache` in
  `api/services/classification_service.py` with
  `cachetools.TTLCache(maxsize=10_000, ttl=3600)` + `@cached`. Entries
  now expire one hour after insertion, in addition to being evicted
  when the size cap is reached.
- A `threading.Lock` is wired in via the `lock=` argument of `@cached`.
  The lock is held only around cache reads and writes, not around the
  wrapped inference call, so concurrent requests still run in parallel.
- Expiry is lazy: stale entries are dropped when their key is next
  accessed or when an insertion scans the cache. There is no background
  sweep, so cache memory is reclaimed on use rather than on a timer.
- Added `cachetools (>=5.3.0,<6.0.0)` as a runtime dependency.

Note: if a worker is still being OOM-killed after this change, the cache
is unlikely to be the cause — at 10,000 entries × ~1–2 KB it is bounded
at roughly 10–20 MB. The more common culprits under sustained
transformers load are PyTorch allocator fragmentation and thread-stack
growth; mitigations there include gunicorn's `--max-requests` to
recycle workers periodically, restricting `OMP_NUM_THREADS`, and
tuning `MALLOC_TRIM_THRESHOLD_`.


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
