# Changelog


## Unreleased

New feature: ATT&CK retrieval with the bi-encoder. Load shedding on the
inference endpoints.

- Every endpoint that runs a model now goes through a per-worker inference
  gate: `ML_GATEWAY_INFERENCE_CONCURRENCY` (default 1) calls run at once,
  `ML_GATEWAY_INFERENCE_QUEUE` (default 32) wait, and the next call is refused
  at once with `503` and `Retry-After: 1`. Previously an overrunning client
  piled up in the request threadpool, every in-flight call spawned its own
  OpenMP threads, and the gateway stopped answering (2026-09-25 outage: 175
  queued requests, 184 threads and four saturated cores per worker).
- `GET /retrieve/attack-biencoder/techniques` and `GET /` run no model and
  answer even while the queue is full.
- `ML_GATEWAY_QUANTIZE=1` runs the severity and attack-technique classifiers
  with dynamic int8 weights, converted in each worker after the fork; about
  twice the throughput at one or two threads per call, responses carry
  `quantized: true`. The bi-encoder is never quantized.
- `GET /stats` reports, per worker, the hit and miss counts of the three
  result caches and the inference gate's limits, occupancy and served /
  refused counts, to tell repeated text from new work under load.
- `ML_GATEWAY_DISABLED_ENDPOINTS` takes the listed endpoints out of service:
  a call is refused with `503` (no `Retry-After`) before any model runs, so
  an operator can shed a whole feature under load.

**Upgrade note:** the server now preloads a new model,
`CIRCL/vulnerability-attack-technique-biencoder`. Run
`poetry run ml-gw-cli refresh-all` (online) once before restarting a
server started with `HF_HUB_OFFLINE=1`, otherwise startup fails with
`LocalEntryNotFoundError`.

- Three new endpoints backed by
  `CIRCL/vulnerability-attack-technique-biencoder`, following VulnTrain's
  `attack-biencoder-retrieval` contract:
  - `POST /index/attack-biencoder` embeds vulnerability descriptions and
    upserts one vector per ID into an on-disk index.
  - `GET /retrieve/attack-biencoder/technique/{id}` ranks indexed
    vulnerabilities for a technique by the training-time probability
    `sigmoid(logit_scale · cosine + logit_bias)`; techniques outside the
    trained vocabulary are embedded from their official ATT&CK text and
    flagged `in_vocabulary: false`.
  - `POST /retrieve/attack-biencoder/related` returns the nearest indexed
    vulnerabilities to an indexed ID or a free text, by plain cosine.
  All responses carry the same `model` / `model_revision` / `error` fields
  as the classification endpoints.
- `POST /index/attack-biencoder` now requires `Authorization: Bearer
  <ML_GATEWAY_INDEX_TOKEN>` (constant-time comparison; `401` otherwise).
  While the variable is unset the endpoint refuses every call with `503`
  and never accepts. The read endpoints stay unauthenticated.
- `ML_GATEWAY_INDEX_MAX_ITEMS` (default 5,000,000) caps the number of
  distinct IDs the index endpoint may grow the index to; a call that
  would exceed it is refused with `507`. Updates of indexed IDs are
  always allowed.
- The README's production example binds `127.0.0.1` and explains why the
  gateway must not be public; docker-compose publishes the port on
  localhost only.
- README reorganised: capabilities overview, endpoints-at-a-glance and
  supported-models tables, model cache and Docker sections, environment
  variables, live example outputs, and a dedicated section on seeding
  the retrieval index.
- No inference runs in the gunicorn master any more. The bi-encoder's
  technique vectors are computed by `AttackBiEncoder.warm_up()` in the
  FastAPI lifespan, i.e. in each worker after the fork: a torch forward
  pass before the fork left the workers with a broken OpenMP thread pool
  whose first inference hung and spun at full CPU.
- New `AttackBiEncoder` wrapper reproducing the scoring function exactly:
  mean pooling over the attention mask, L2 normalization, 512-token
  vulnerability texts and `technique_max_length` technique texts, affine
  constants and trained technique texts read from the model release.
- New `VectorStore`: an append-only float16 matrix and ID list on disk,
  memory-mapped so all gunicorn workers share one copy and see each
  other's appends, brute-force cosine search, advisory file lock for
  writers, torn-write repair, and a `meta.json` pinning the index to one
  model revision (requests report an error asking for a rebuild when the
  served model changes). Location: `ML_GATEWAY_INDEX_DIR` (default
  `./index`); docker-compose mounts a named volume for it.
- Bundled `api/data/attack_technique_texts.json`: `"Name. Description"`
  for every active enterprise ATT&CK technique (STIX bundle v19.1, markup
  stripped), identical to the model's shipped texts for the 53 trained
  techniques.
- Two CLI commands fill the index in bulk, both safe to run while the
  server is up:
  - `ml-gw-cli backfill-index --dumps <file-or-dir>…` reads
    Vulnerability-Lookup's NDJSON feed dumps (plain or gzipped; CVE JSON 5,
    NVD API, OSV, CSAF, JVNDB and VARIoT layouts), embeds each description
    on the gateway host in length-sorted batches, indexes each ID once and
    prints a per-feed report; `--skip-existing` resumes an interrupted
    run, `--limit` caps it.
  - `ml-gw-cli embed-dumps --dumps … --output vectors.npz --device cuda`
    runs the same extraction on a GPU host and writes the vectors to an
    `.npz` archive (`ids`, float16 `embeddings`, `model`,
    `model_revision`) instead of an index. `AttackBiEncoder` accepts a
    torch device for this; the server keeps using the CPU.
  - `ml-gw-cli import-index --file vectors.npz` imports such an archive,
    refusing one whose revision differs from the served model or whose
    vectors are not L2-normalized.
- `ml-gw-cli refresh-all` now iterates over the model registries (so the
  bi-encoder and its `technique_texts.json` are cached at build time)
  instead of a hard-coded list.
- `numpy` and `huggingface-hub` are declared as direct dependencies.
- New test suites for the vector store and the retrieval endpoints
  (stubbed encoder, real store in a temporary directory).


## Release 1.4.0 (2026-08-30)

Fix for long inputs crashing inference, plus documentation and
dependency updates.

- Fixed a crash on long descriptions: some fine-tuned repos (notably the
  Chinese MacBERT model) ship no `model_max_length` in their
  `tokenizer_config.json`, so transformers reports a huge sentinel value
  and `truncation=True` never actually truncated — inputs over 512
  tokens overflowed the model's position-embedding table at inference
  time. Both classifiers now clamp the tokenizer's limit to the model's
  `max_position_embeddings` (minus the two offset slots RoBERTa-style
  models reserve) right after loading, via a shared
  `clamp_tokenizer_max_length` helper with its own test suite.
- Documentation: the README now shows `HF_HUB_OFFLINE=1` as an example
  of starting the server without pulling model updates from the
  Hugging Face Hub.
- Updated dependencies.


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
- New test suite (`poetry run pytest`) covering both classification
  endpoints through the router and service layers with a stubbed model
  layer, so it runs without downloading models. `pytest` and `httpx` are
  added as a `dev` dependency group.
- New CI workflow (GitHub Actions) running mypy and the test suite on
  every push to `main` and every pull request.
- The classification endpoints are now plain (non-async) handlers, so
  FastAPI runs them in its threadpool and CPU-bound model inference no
  longer blocks the event loop for concurrent requests.
- `mypy` and `types-cachetools` are now dev dependencies, so
  `poetry run mypy api/` uses the project virtualenv instead of relying
  on a system-wide mypy that cannot see the dependencies;
  `explicit_package_bases` is enabled since `api/` is a namespace
  package. Fixed the one strictness error this surfaced
  (`Tensor.item()` is typed `int | float`; the argmax index is now
  wrapped in `int()`).


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
