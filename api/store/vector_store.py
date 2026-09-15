import fcntl
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

"""
On-disk vector index shared by every worker process of the gateway.

The index is a directory holding an append-only float16 matrix
(``vectors.f16``, one row per upsert), the IDs in the same order
(``ids.txt``, one per line) and a ``meta.json`` recording the model and
revision the vectors were computed with. Search is brute force: the matrix
is memory-mapped, so with several gunicorn workers the operating system
keeps a single copy in the page cache, and every worker picks up rows
appended by another one before each search by re-reading the tail of the
files. Appends are serialized across processes with an advisory lock.

An upsert for an ID that is already indexed appends a new row; the latest
row for an ID is the live one and superseded rows are masked out of every
search. Writers write the ID line before the vector row, so a reader never
sees a vector without its ID; a torn write from a crashed writer leaves a
surplus tail that the next writer truncates before appending.
"""

_FORMAT_VERSION = 1
_DTYPE = np.float16
# Rows scored per step of the brute-force search. Each step converts one
# float16 slice of the memory-mapped matrix to float32 (the BLAS-friendly
# type) — 64 k rows × 768 dims is a 200 MB temporary, small enough to keep
# memory flat while large enough to amortize the per-call overhead.
_SEARCH_CHUNK_ROWS = 65_536


class IndexRevisionMismatch(ValueError):
    """The on-disk index was built with a different model or revision."""


class VectorStore:
    """Append-only, memory-mapped cosine index for one model revision.

    Vectors are expected to be L2-normalized by the caller; the store
    computes plain dot products.
    """

    def __init__(
        self,
        directory: Path,
        dimension: int,
        model: str,
        model_revision: str | None,
    ) -> None:
        self.directory = directory
        self.dimension = dimension
        self.model = model
        self.model_revision = model_revision
        self._row_bytes = dimension * np.dtype(_DTYPE).itemsize
        self._ids_path = directory / "ids.txt"
        self._vectors_path = directory / "vectors.f16"
        self._meta_path = directory / "meta.json"
        self._lock_path = directory / ".lock"
        # Guards the in-memory view within one process; ``_file_lock``
        # serializes writers across processes.
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._row_of: dict[str, int] = {}
        self._active: NDArray[np.bool_] = np.zeros(0, dtype=np.bool_)
        self._matrix: NDArray[np.float16] | None = None
        # Bytes of ``ids.txt`` already consumed into ``_ids``.
        self._ids_offset = 0

        directory.mkdir(parents=True, exist_ok=True)
        with self._lock, self._file_lock():
            self._check_meta()
            self._ids_path.touch()
            self._vectors_path.touch()
            self._repair_tail()

    # -- metadata -----------------------------------------------------------

    def _check_meta(self) -> None:
        if not self._meta_path.exists():
            meta = {
                "format": _FORMAT_VERSION,
                "model": self.model,
                "model_revision": self.model_revision,
                "dimension": self.dimension,
                "dtype": np.dtype(_DTYPE).name,
            }
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)
            return
        with open(self._meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("model") != self.model or meta.get("model_revision") != self.model_revision:
            raise IndexRevisionMismatch(
                f"Index at {self.directory} was built with {meta.get('model')} "
                f"revision {meta.get('model_revision')}, but the served model is "
                f"{self.model} revision {self.model_revision}. Vectors are only "
                "comparable within one revision: rebuild the index."
            )
        if int(meta.get("dimension", -1)) != self.dimension:
            raise ValueError(
                f"Index at {self.directory} has dimension {meta.get('dimension')}, "
                f"the served model produces {self.dimension}."
            )

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        with open(self._lock_path, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # -- reading ------------------------------------------------------------

    def _refresh(self) -> None:
        """Pull rows appended since the last refresh into the in-memory view."""
        rows_on_disk = self._vectors_path.stat().st_size // self._row_bytes
        loaded = len(self._ids)
        if rows_on_disk <= loaded:
            return
        with open(self._ids_path, "rb") as f:
            f.seek(self._ids_offset)
            tail = f.read()
        # The last element is either empty or a line still being written.
        lines = tail.split(b"\n")[:-1][: rows_on_disk - loaded]
        if not lines:
            return
        new_ids = [line.decode("utf-8") for line in lines]
        self._ids_offset += sum(len(line) + 1 for line in lines)
        # A fresh mask each refresh: searches snapshot the previous one
        # outside the lock, so it must never be mutated in place.
        active = np.concatenate([self._active, np.ones(len(new_ids), dtype=np.bool_)])
        for row, id_ in enumerate(new_ids, start=loaded):
            previous = self._row_of.get(id_)
            if previous is not None:
                active[previous] = False
            self._row_of[id_] = row
        self._ids.extend(new_ids)
        self._active = active
        self._matrix = np.memmap(
            self._vectors_path,
            dtype=_DTYPE,
            mode="r",
            shape=(len(self._ids), self.dimension),
        )

    @property
    def count(self) -> int:
        """Number of live IDs (superseded rows excluded)."""
        with self._lock:
            self._refresh()
            return int(self._active.sum())

    def get(self, id_: str) -> NDArray[np.float32] | None:
        """Return the live vector stored for ``id_``, or ``None``."""
        with self._lock:
            self._refresh()
            row = self._row_of.get(id_)
            matrix = self._matrix
        if row is None or matrix is None:
            return None
        return np.asarray(matrix[row], dtype=np.float32)

    def search(
        self, query: NDArray[np.float32], top_k: int, exclude: str | None = None
    ) -> list[tuple[str, float]]:
        """Top-k live IDs by dot product with ``query``, best first."""
        with self._lock:
            self._refresh()
            matrix, active, ids = self._matrix, self._active, self._ids
            exclude_row = self._row_of.get(exclude) if exclude is not None else None
        if matrix is None or top_k <= 0:
            return []
        rows = matrix.shape[0]
        vector = np.asarray(query, dtype=np.float32)
        scores = np.empty(rows, dtype=np.float32)
        for start in range(0, rows, _SEARCH_CHUNK_ROWS):
            stop = min(start + _SEARCH_CHUNK_ROWS, rows)
            scores[start:stop] = np.asarray(matrix[start:stop], dtype=np.float32) @ vector
        scores[~active[:rows]] = -np.inf
        if exclude_row is not None and exclude_row < rows:
            scores[exclude_row] = -np.inf
        k = min(top_k, rows)
        best = np.argpartition(-scores, k - 1)[:k]
        best = best[np.argsort(-scores[best], kind="stable")]
        return [
            (ids[int(row)], float(scores[row])) for row in best if np.isfinite(scores[row])
        ]

    # -- writing ------------------------------------------------------------

    def _truncate(self, path: Path, size: int) -> None:
        with open(path, "r+b") as f:
            f.truncate(size)

    def _repair_tail(self) -> None:
        """Drop torn tails left by a writer that died mid-append.

        Must be called with both locks held. A partial vector row is cut,
        then IDs without a vector (IDs are written first) and vectors
        without an ID are truncated so both files line up again.
        """
        vectors_size = self._vectors_path.stat().st_size
        rows = vectors_size // self._row_bytes
        if vectors_size != rows * self._row_bytes:
            self._truncate(self._vectors_path, rows * self._row_bytes)
        self._refresh()
        if self._ids_path.stat().st_size > self._ids_offset:
            self._truncate(self._ids_path, self._ids_offset)
        if rows > len(self._ids):
            self._truncate(self._vectors_path, len(self._ids) * self._row_bytes)

    def upsert(self, ids: list[str], vectors: NDArray[np.float32]) -> None:
        """Append one row per ID; a repeated ID supersedes its earlier row."""
        if vectors.ndim != 2 or vectors.shape != (len(ids), self.dimension):
            raise ValueError(
                f"Expected {len(ids)} vectors of dimension {self.dimension}, "
                f"got an array of shape {vectors.shape}."
            )
        if any("\n" in id_ or not id_ for id_ in ids):
            raise ValueError("IDs must be non-empty and must not contain newlines.")
        if not ids:
            return
        lines = "".join(f"{id_}\n" for id_ in ids).encode("utf-8")
        data = np.ascontiguousarray(vectors, dtype=_DTYPE).tobytes()
        with self._lock, self._file_lock():
            self._repair_tail()
            for path, payload in ((self._ids_path, lines), (self._vectors_path, data)):
                with open(path, "ab") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
            self._refresh()
