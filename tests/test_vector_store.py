from pathlib import Path

import numpy as np
import pytest

from api.store.vector_store import IndexRevisionMismatch, VectorStore

"""
Unit tests for the on-disk vector index: append/upsert semantics, cross-
process visibility (modelled with two store instances on one directory),
revision pinning and recovery from torn writes.
"""

MODEL = "stub/biencoder"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DIM = 4


def unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def open_store(directory: Path, revision: str | None = REVISION) -> VectorStore:
    return VectorStore(directory, DIM, MODEL, revision)


def test_empty_store(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    assert store.count == 0
    assert store.search(unit(1, 0, 0, 0), top_k=5) == []
    assert store.get("CVE-1") is None


def test_search_ranks_by_cosine(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.upsert(
        ["a", "b", "c"],
        np.stack([unit(1, 0, 0, 0), unit(1, 1, 0, 0), unit(0, 1, 0, 0)]),
    )
    hits = store.search(unit(1, 0, 0, 0), top_k=2)
    assert [id_ for id_, _ in hits] == ["a", "b"]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-3)
    assert hits[1][1] == pytest.approx(0.7071, abs=1e-3)
    assert store.count == 3


def test_top_k_larger_than_index(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.upsert(["a"], np.stack([unit(1, 0, 0, 0)]))
    assert len(store.search(unit(1, 0, 0, 0), top_k=50)) == 1


def test_upsert_supersedes_previous_row(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.upsert(["a", "b"], np.stack([unit(1, 0, 0, 0), unit(0, 1, 0, 0)]))
    store.upsert(["a"], np.stack([unit(0, 0, 1, 0)]))
    assert store.count == 2
    # The old x-axis vector for "a" is gone: nothing scores 1.0 on x any more.
    assert max(score for _, score in store.search(unit(1, 0, 0, 0), top_k=5)) < 0.5
    # The new one wins on the z axis.
    assert store.search(unit(0, 0, 1, 0), top_k=1)[0][0] == "a"
    vector = store.get("a")
    assert vector is not None
    assert np.allclose(vector, unit(0, 0, 1, 0), atol=1e-3)


def test_exclude_id(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.upsert(["a", "b"], np.stack([unit(1, 0, 0, 0), unit(1, 0.1, 0, 0)]))
    hits = store.search(unit(1, 0, 0, 0), top_k=5, exclude="a")
    assert [id_ for id_, _ in hits] == ["b"]


def test_appends_are_visible_to_another_instance(tmp_path: Path) -> None:
    writer = open_store(tmp_path)
    reader = open_store(tmp_path)
    assert reader.count == 0
    writer.upsert(["a"], np.stack([unit(1, 0, 0, 0)]))
    assert reader.count == 1
    assert reader.search(unit(1, 0, 0, 0), top_k=1)[0][0] == "a"
    writer.upsert(["a", "b"], np.stack([unit(0, 1, 0, 0), unit(1, 0, 0, 0)]))
    assert [id_ for id_, _ in reader.search(unit(1, 0, 0, 0), top_k=5)] == ["b", "a"]
    assert reader.count == 2


def test_reopen_restores_index(tmp_path: Path) -> None:
    open_store(tmp_path).upsert(["a", "a", "b"], np.stack([unit(1, 0, 0, 0)] * 3))
    reopened = open_store(tmp_path)
    assert reopened.count == 2
    assert sorted(id_ for id_, _ in reopened.search(unit(1, 0, 0, 0), top_k=5)) == ["a", "b"]


def test_revision_mismatch_is_refused(tmp_path: Path) -> None:
    open_store(tmp_path)
    with pytest.raises(IndexRevisionMismatch, match="rebuild the index"):
        open_store(tmp_path, revision="f" * 40)
    with pytest.raises(ValueError, match="dimension"):
        VectorStore(tmp_path, DIM + 1, MODEL, REVISION)


def test_torn_tail_is_repaired(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    store.upsert(["a"], np.stack([unit(1, 0, 0, 0)]))
    # A writer that died after the ID line and half a vector row.
    with open(tmp_path / "ids.txt", "ab") as f:
        f.write(b"orphan\n")
    with open(tmp_path / "vectors.f16", "ab") as f:
        f.write(b"\x00" * 3)
    reopened = open_store(tmp_path)
    assert reopened.count == 1
    reopened.upsert(["b"], np.stack([unit(0, 1, 0, 0)]))
    assert reopened.search(unit(0, 1, 0, 0), top_k=1)[0][0] == "b"
    assert (tmp_path / "ids.txt").read_bytes() == b"a\nb\n"


def test_rejects_malformed_upserts(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        store.upsert(["a"], np.zeros((1, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError, match="newlines"):
        store.upsert(["a\nb"], np.zeros((1, DIM), dtype=np.float32))
    store.upsert([], np.zeros((0, DIM), dtype=np.float32))
    assert store.count == 0
