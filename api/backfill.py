import gzip
import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from api.models.biencoder_model import AttackBiEncoder
from api.store.vector_store import VectorStore

"""
Bulk population of the bi-encoder index, for the ``ml-gw-cli`` commands.

Two paths feed the same store the ``POST /index/attack-biencoder`` endpoint
writes to: a backfill that reads Vulnerability-Lookup's NDJSON feed dumps
and embeds every description on this host, and an import of vectors
computed elsewhere (a GPU box running the reference snippet) shipped as one
``.npz`` file. Both are safe to run while the server is up — the store's
file lock serializes the appends and the workers pick them up.
"""

# Per-line metadata the dump writer adds next to the raw record.
_META_PREFIX = "vulnerability-lookup:"
# Dumps that are not vulnerability feeds.
_EXCLUDED_DUMPS = {"comments", "bundles", "sightings", "kev_entries"}


def _english(descriptions: Any) -> str | None:
    """First English ``{"lang", "value"}`` entry, the rule Vulnerability-Lookup
    itself uses for the description it shows (and sends to the gateway)."""
    if not isinstance(descriptions, list):
        return None
    for description in descriptions:
        if not isinstance(description, dict):
            continue
        value = description.get("value")
        if value and str(description.get("lang", "")).lower().startswith("en"):
            return str(value)
    return None


def _csaf_summary(record: dict[str, Any]) -> str | None:
    """Summary notes of every vulnerability, else the document summary."""
    parts = [
        note["text"]
        for vulnerability in record.get("vulnerabilities") or []
        if isinstance(vulnerability, dict)
        for note in vulnerability.get("notes") or []
        if isinstance(note, dict) and note.get("category") == "summary" and note.get("text")
    ]
    if parts:
        return " ".join(parts)
    for note in record["document"].get("notes") or []:
        if isinstance(note, dict) and note.get("category") == "summary" and note.get("text"):
            return str(note["text"])
    return None


def extract(record: Any) -> tuple[str, str] | None:
    """``(id, description)`` for one dump record, or ``None`` if it has none.

    Recognizes the feed layouts Vulnerability-Lookup stores: CVE JSON 5
    (cvelistv5, nvd, GNA and GCVE pulls), the NVD API shape (fkie_nvd), OSV
    (GitHub, PySec, OSSF), CSAF, flat records with an ``id`` and a
    ``description`` string (JVNDB and similar) and VARIoT's ``description.data``. The ID is the public
    vulnerability ID, i.e. the one Vulnerability-Lookup pages link to.
    """
    if not isinstance(record, dict):
        return None
    id_: Any = None
    text: str | None = None
    if isinstance(record.get("containers"), dict):
        metadata = record.get("cveMetadata")
        id_ = metadata.get("cveId") if isinstance(metadata, dict) else None
        id_ = id_ or record.get(f"{_META_PREFIX}id")
        cna = record["containers"].get("cna")
        text = _english(cna.get("descriptions")) if isinstance(cna, dict) else None
    elif isinstance(record.get("document"), dict):
        tracking = record["document"].get("tracking")
        id_ = tracking.get("id") if isinstance(tracking, dict) else None
        text = _csaf_summary(record)
    elif isinstance(record.get("descriptions"), list):
        id_ = record.get("id")
        text = _english(record["descriptions"])
    elif isinstance(record.get("details"), str):
        id_ = record.get("id")
        text = record["details"]
    elif isinstance(record.get("description"), (str, dict)):
        id_ = record.get("id") or record.get("sec:identifier") or record.get(f"{_META_PREFIX}id")
        description = record["description"]
        # VARIoT wraps the text as ``{"data": ..., "sources": [...]}``.
        text = description if isinstance(description, str) else description.get("data")
        if not isinstance(text, str):
            return None
    else:
        return None
    if not id_ or not text:
        return None
    id_ = str(id_).strip()
    text = text.strip()
    if not id_ or not text or any(char.isspace() for char in id_):
        return None
    return id_, text


def resolve_dump_paths(paths: Iterable[Path]) -> list[Path]:
    """Expand directories to their feed dumps (sorted), keep files as given.

    Sorting matters when one vulnerability appears in several feeds: the
    backfill keeps the first occurrence, so ``cvelistv5`` wins over
    ``fkie_nvd`` and ``nvd`` for a CVE.
    """
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.iterdir()):
                name = candidate.name
                if name.endswith(".ndjson.gz"):
                    stem = name[: -len(".ndjson.gz")]
                elif name.endswith(".ndjson"):
                    stem = name[: -len(".ndjson")]
                else:
                    continue
                if stem not in _EXCLUDED_DUMPS:
                    resolved.append(candidate)
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"No such dump file or directory: {path}")
    return resolved


def iter_records(paths: Iterable[Path]) -> Iterator[tuple[Path, int, Any]]:
    """Yield ``(path, line_number, record)``; a malformed line yields ``None``."""
    for path in paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, number, json.loads(line)
                except json.JSONDecodeError:
                    yield path, number, None


@dataclass
class BackfillReport:
    files: int = 0
    records: int = 0
    indexed: int = 0
    duplicates: int = 0
    skipped_existing: int = 0
    unusable: int = 0
    malformed: int = 0
    # Records without a usable ID/description, per dump — feeds with no
    # recognized layout show up here as a whole.
    unusable_by_file: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"files: {self.files}",
            f"records: {self.records}",
            f"indexed: {self.indexed}",
            f"duplicates (later occurrences of an ID): {self.duplicates}",
            f"skipped (already indexed): {self.skipped_existing}",
            f"unusable (no ID or description): {self.unusable}",
            f"malformed lines: {self.malformed}",
        ]
        for name, count in sorted(self.unusable_by_file.items()):
            lines.append(f"  unusable in {name}: {count}")
        return "\n".join(lines)


def backfill(
    encoder: AttackBiEncoder,
    store: VectorStore,
    paths: Iterable[Path],
    batch_size: int = 64,
    skip_existing: bool = False,
    limit: int | None = None,
    progress: Callable[[BackfillReport], None] | None = None,
    progress_every: int = 1000,
) -> BackfillReport:
    """Embed every usable record of the dumps and upsert it into the store.

    One ID is indexed at most once per run (first occurrence wins); with
    ``skip_existing`` IDs already in the store are left alone, which makes
    an interrupted run resumable. ``limit`` caps the number of indexed
    records.
    """
    files = resolve_dump_paths(paths)
    report = BackfillReport(files=len(files))
    seen: set[str] = set()
    buffer: list[tuple[str, str]] = []
    # Descriptions are padded to the longest text of their batch, so one
    # long text makes a whole batch cost as much as 512-token inputs. Buffer
    # several batches and sort by length so each batch is homogeneous.
    buffer_size = batch_size * 8

    def flush() -> None:
        buffer.sort(key=lambda item: len(item[1]))
        for start in range(0, len(buffer), batch_size):
            batch = buffer[start : start + batch_size]
            ids = [id_ for id_, _ in batch]
            store.upsert(ids, encoder.embed_vulnerabilities([text for _, text in batch]))
            report.indexed += len(ids)
        buffer.clear()

    for path, _, record in iter_records(files):
        if limit is not None and report.indexed + len(buffer) >= limit:
            break
        report.records += 1
        if progress is not None and report.records % progress_every == 0:
            progress(report)
        if record is None:
            report.malformed += 1
            continue
        extracted = extract(record)
        if extracted is None:
            report.unusable += 1
            report.unusable_by_file[path.name] = report.unusable_by_file.get(path.name, 0) + 1
            continue
        id_, text = extracted
        if id_ in seen:
            report.duplicates += 1
            continue
        seen.add(id_)
        if skip_existing and store.contains(id_):
            report.skipped_existing += 1
            continue
        buffer.append((id_, text))
        if len(buffer) >= buffer_size:
            flush()
    flush()
    return report


def embed_dumps(
    encoder: AttackBiEncoder,
    paths: Iterable[Path],
    output: Path,
    batch_size: int = 64,
    limit: int | None = None,
    progress: Callable[[BackfillReport], None] | None = None,
    progress_every: int = 1000,
) -> BackfillReport:
    """Embed the dumps into one ``.npz`` for :func:`import_vectors`.

    The counterpart of :func:`backfill` for a host that has a GPU but no
    index: same extraction, same batching, but the vectors go to an
    archive (``ids``, float16 ``embeddings``, ``model`` and
    ``model_revision``) instead of the store. ``report.indexed`` counts the
    vectors written.
    """
    ids: list[str] = []
    vectors: list[NDArray[np.float32]] = []

    class _Sink:
        """The slice of the store interface ``backfill`` writes to."""

        dimension = encoder.dimension

        def contains(self, id_: str) -> bool:
            return False

        def upsert(self, batch_ids: list[str], batch_vectors: NDArray[np.float32]) -> None:
            ids.extend(batch_ids)
            vectors.append(batch_vectors)

    report = backfill(
        encoder,
        _Sink(),  # type: ignore[arg-type]
        paths,
        batch_size=batch_size,
        limit=limit,
        progress=progress,
        progress_every=progress_every,
    )
    embeddings = (
        np.concatenate(vectors).astype(np.float16)
        if vectors
        else np.zeros((0, encoder.dimension), dtype=np.float16)
    )
    np.savez(
        output,
        ids=np.array(ids),
        embeddings=embeddings,
        model=encoder.model_name,
        model_revision=encoder.revision or "",
    )
    return report


def _scalar(value: NDArray[Any]) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def import_vectors(
    store: VectorStore,
    path: Path,
    model: str,
    model_revision: str | None,
    batch_size: int = 10_000,
) -> int:
    """Upsert vectors computed elsewhere, shipped as one ``.npz``.

    The archive carries ``ids`` (array of strings), ``embeddings`` (N × dim,
    float16 or wider) and ``model_revision``; an optional ``model`` must
    match too. Vectors must be L2-normalized already, as the reference
    snippet produces them — the import refuses anything else rather than
    silently re-normalizing a wrongly computed matrix. Returns the number
    of vectors imported.
    """
    with np.load(path, allow_pickle=False) as archive:
        for key in ("ids", "embeddings", "model_revision"):
            if key not in archive:
                raise ValueError(f"{path}: missing '{key}' in the archive.")
        revision = _scalar(archive["model_revision"])
        if revision != (model_revision or ""):
            raise ValueError(
                f"{path}: vectors were computed with revision {revision}, but the served "
                f"model is revision {model_revision}. Only vectors from the same revision "
                "can be imported."
            )
        if "model" in archive and _scalar(archive["model"]) != model:
            raise ValueError(
                f"{path}: vectors were computed with {_scalar(archive['model'])}, "
                f"but the served model is {model}."
            )
        ids = [str(id_) for id_ in np.asarray(archive["ids"]).reshape(-1)]
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape != (len(ids), store.dimension):
        raise ValueError(
            f"{path}: expected embeddings of shape ({len(ids)}, {store.dimension}), "
            f"got {embeddings.shape}."
        )
    norms = np.linalg.norm(embeddings, axis=1)
    if len(norms) and not np.allclose(norms, 1.0, atol=1e-2):
        raise ValueError(f"{path}: embeddings are not L2-normalized (norms range {norms.min():.3f}–{norms.max():.3f}).")
    for start in range(0, len(ids), batch_size):
        store.upsert(ids[start : start + batch_size], embeddings[start : start + batch_size])
    return len(ids)
