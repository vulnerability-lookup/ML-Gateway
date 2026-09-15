import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from api.backfill import backfill, extract, import_vectors, iter_records, resolve_dump_paths
from api.store.vector_store import VectorStore

"""
Tests for the dump backfill and the .npz import: record extraction for
every feed layout Vulnerability-Lookup dumps, dump discovery, the backfill
loop (stub encoder, real store) and the import's revision and shape checks.
"""

MODEL = "stub/biencoder"
REVISION = "0123456789abcdef0123456789abcdef01234567"
DIM = 4

CVE5 = {
    "dataType": "CVE_RECORD",
    "cveMetadata": {"cveId": "CVE-1999-0001"},
    "containers": {
        "cna": {
            "descriptions": [
                {"lang": "es", "value": "ip_input.c en implementaciones…"},
                {"lang": "en", "value": "ip_input.c in BSD-derived TCP/IP implementations…"},
            ]
        }
    },
    "vulnerability-lookup:id": "cve-1999-0001",
}
FKIE = {
    "id": "CVE-1999-0001",
    "descriptions": [{"lang": "en", "value": "ip_input.c (NVD copy)"}],
    "metrics": {},
}
OSV = {"id": "GHSA-2222-76gx-28mm", "details": "A origin validation error…", "aliases": []}
CSAF = {
    "document": {
        "tracking": {"id": "RHBA-2005:001"},
        "notes": [{"category": "summary", "text": "Document summary."}],
    },
    "vulnerabilities": [
        {"notes": [{"category": "summary", "text": "Kernel flaw."}, {"category": "other", "text": "x"}]},
        {"notes": [{"category": "summary", "text": "Second flaw."}]},
    ],
}
JVNDB = {"sec:identifier": "JVNDB-2002-000291", "description": "Canna contains a buffer overflow."}
VARIOT = {"id": "VAR-190001-0018", "description": {"sources": [{"db": "CNVD"}], "data": "SAP NetWeaver flaw."}}


@pytest.mark.parametrize(
    "record, expected",
    [
        (CVE5, ("CVE-1999-0001", "ip_input.c in BSD-derived TCP/IP implementations…")),
        (FKIE, ("CVE-1999-0001", "ip_input.c (NVD copy)")),
        (OSV, ("GHSA-2222-76gx-28mm", "A origin validation error…")),
        (CSAF, ("RHBA-2005:001", "Kernel flaw. Second flaw.")),
        ({"document": CSAF["document"], "vulnerabilities": []}, ("RHBA-2005:001", "Document summary.")),
        (JVNDB, ("JVNDB-2002-000291", "Canna contains a buffer overflow.")),
        (VARIOT, ("VAR-190001-0018", "SAP NetWeaver flaw.")),
        # GNA/GCVE pulls may lack cveMetadata: fall back to the dump's own ID.
        ({**CVE5, "cveMetadata": {}}, ("cve-1999-0001", "ip_input.c in BSD-derived TCP/IP implementations…")),
    ],
)
def test_extract_known_layouts(record: dict[str, Any], expected: tuple[str, str]) -> None:
    assert extract(record) == expected


@pytest.mark.parametrize(
    "record",
    [
        None,
        "not a dict",
        {"id": "X", "title": "no description"},
        {"containers": {"cna": {"descriptions": [{"lang": "fr", "value": "…"}]}}, "cveMetadata": {"cveId": "CVE-1"}},
        {"id": "with space", "details": "text"},
        {"id": "", "details": "text"},
        {"id": "X", "details": "   "},
        {"id": "VAR-1", "description": {"sources": []}},
    ],
)
def test_extract_rejects_unusable_records(record: Any) -> None:
    assert extract(record) is None


def write_ndjson(path: Path, records: list[Any], compress: bool = False) -> Path:
    payload = "".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in records)
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    return path


def test_resolve_dump_paths_and_iteration(tmp_path: Path) -> None:
    write_ndjson(tmp_path / "cvelistv5.ndjson", [CVE5])
    write_ndjson(tmp_path / "github.ndjson.gz", [OSV, "{not json"], compress=True)
    write_ndjson(tmp_path / "comments.ndjson", [{"id": "c"}])
    (tmp_path / "notes.txt").write_text("ignored")
    paths = resolve_dump_paths([tmp_path])
    assert [p.name for p in paths] == ["cvelistv5.ndjson", "github.ndjson.gz"]
    records = list(iter_records(paths))
    assert [(p.name, n) for p, n, _ in records] == [
        ("cvelistv5.ndjson", 1),
        ("github.ndjson.gz", 1),
        ("github.ndjson.gz", 2),
    ]
    assert records[-1][2] is None
    with pytest.raises(FileNotFoundError):
        resolve_dump_paths([tmp_path / "missing.ndjson"])


class StubBiEncoder:
    model_name = MODEL
    revision = REVISION
    dimension = DIM

    def __init__(self) -> None:
        self.calls: list[int] = []

    def embed_vulnerabilities(self, descriptions: list[str]) -> NDArray[np.float32]:
        self.calls.append(len(descriptions))
        vectors = np.zeros((len(descriptions), DIM), dtype=np.float32)
        for row, text in enumerate(descriptions):
            vectors[row, len(text) % DIM] = 1.0
        return vectors


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "index", DIM, MODEL, REVISION)


def test_backfill_indexes_first_occurrence_and_batches(tmp_path: Path, store: VectorStore) -> None:
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    write_ndjson(dumps / "cvelistv5.ndjson", [CVE5, {"id": "X", "title": "unusable"}, "{broken"])
    write_ndjson(dumps / "fkie_nvd.ndjson", [FKIE])  # same CVE, later feed: a duplicate
    write_ndjson(dumps / "github.ndjson", [OSV, JVNDB])
    encoder = StubBiEncoder()
    seen: list[int] = []

    report = backfill(
        encoder, store, [dumps], batch_size=2,
        progress=lambda r: seen.append(r.records), progress_every=2,
    )
    assert (report.files, report.records, report.indexed) == (3, 6, 3)
    assert (report.duplicates, report.unusable, report.malformed) == (1, 1, 1)
    assert report.unusable_by_file == {"cvelistv5.ndjson": 1}
    assert encoder.calls == [2, 1]
    assert seen == [2, 4, 6]
    assert store.count == 3
    # The cvelistv5 description won over the fkie_nvd copy for the CVE.
    cve5_text = CVE5["containers"]["cna"]["descriptions"][1]["value"]  # type: ignore[index]
    expected = np.zeros(DIM, dtype=np.float32)
    expected[len(cve5_text) % DIM] = 1.0
    assert np.allclose(store.get("CVE-1999-0001"), expected)
    assert "indexed: 3" in report.summary()


def test_backfill_skip_existing_and_limit(tmp_path: Path, store: VectorStore) -> None:
    dump = write_ndjson(tmp_path / "feed.ndjson", [CVE5, OSV, JVNDB])
    encoder = StubBiEncoder()
    store.upsert(["CVE-1999-0001"], np.eye(DIM, dtype=np.float32)[:1])

    report = backfill(encoder, store, [dump], skip_existing=True, limit=1)
    assert report.skipped_existing == 1
    assert report.indexed == 1
    assert store.count == 2
    assert store.contains("GHSA-2222-76gx-28mm")
    assert not store.contains("JVNDB-2002-000291")

    report = backfill(encoder, store, [dump])
    assert (report.indexed, report.skipped_existing) == (3, 0)
    assert store.count == 3


def make_npz(path: Path, ids: list[str], embeddings: NDArray[Any], **extra: Any) -> Path:
    np.savez(path, ids=np.array(ids), embeddings=embeddings, model_revision=REVISION, **extra)
    return path


def test_import_vectors(tmp_path: Path, store: VectorStore) -> None:
    embeddings = np.eye(DIM, dtype=np.float16)[:2]
    archive = make_npz(tmp_path / "vectors.npz", ["CVE-1", "CVE-2"], embeddings, model=MODEL)
    assert import_vectors(store, archive, MODEL, REVISION, batch_size=1) == 2
    assert store.count == 2
    assert store.search(np.eye(DIM, dtype=np.float32)[1], top_k=1)[0][0] == "CVE-2"


@pytest.mark.parametrize(
    "ids, embeddings, extra, served_revision, message",
    [
        (["CVE-1"], np.eye(DIM, dtype=np.float16)[:1], {}, "f" * 40, "revision"),
        (["CVE-1"], np.eye(DIM, dtype=np.float16)[:1], {"model": "other/model"}, REVISION, "other/model"),
        (["CVE-1", "CVE-2"], np.eye(DIM, dtype=np.float16)[:1], {}, REVISION, "shape"),
        (["CVE-1"], np.eye(DIM + 1, dtype=np.float16)[:1], {}, REVISION, "shape"),
        (["CVE-1"], 2 * np.eye(DIM, dtype=np.float16)[:1], {}, REVISION, "normalized"),
    ],
)
def test_import_refuses_bad_archives(
    tmp_path: Path, store: VectorStore, ids: list[str], embeddings: NDArray[Any],
    extra: dict[str, Any], served_revision: str, message: str,
) -> None:
    archive = make_npz(tmp_path / "vectors.npz", ids, embeddings, **extra)
    with pytest.raises(ValueError, match=message):
        import_vectors(store, archive, MODEL, served_revision)
    assert store.count == 0


def test_import_requires_the_contract_keys(tmp_path: Path, store: VectorStore) -> None:
    archive = tmp_path / "vectors.npz"
    np.savez(archive, ids=np.array(["CVE-1"]), embeddings=np.eye(DIM, dtype=np.float16)[:1])
    with pytest.raises(ValueError, match="model_revision"):
        import_vectors(store, archive, MODEL, REVISION)
