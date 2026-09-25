from pathlib import Path

import typer
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from api.backfill import BackfillReport, backfill, embed_dumps, import_vectors
from api.models.attack_model import ATTACK_MODELS, get_attack_model_instance
from api.models.biencoder_model import BIENCODER_MODELS, AttackBiEncoder, get_biencoder_instance
from api.models.quantization import cpu_flags, select_engine
from api.models.revisions import pinned_revision
from api.models.severity_model import LABELS, get_model_instance
from api.schemas import DEFAULT_BIENCODER_MODEL
from api.services.retrieval_service import get_store

app = typer.Typer(help="Utility CLI for managing NLP models.")


def _refresh(model_name: str, revision: str | None = None) -> None:
    """Force-download one model's tokenizer, weights and companion files.

    ``revision`` is a commit SHA (or branch/tag); ``None`` means the pin from
    ``ML_GATEWAY_MODEL_REVISIONS`` if any, else the Hub's ``main``. The same
    rule the server applies, so what gets cached is what gets served.
    """
    revision = revision or pinned_revision(model_name)
    typer.echo(f"Refreshing model: {model_name} (revision {revision or 'main'})")
    try:
        _ = AutoTokenizer.from_pretrained(model_name, revision=revision, force_download=True)
        if model_name in BIENCODER_MODELS:
            # A plain encoder, shipped with the technique texts it was
            # trained against; the server reads both from the cache.
            _ = AutoModel.from_pretrained(model_name, revision=revision, force_download=True)
            _ = hf_hub_download(model_name, "technique_texts.json", revision=revision, force_download=True)
        else:
            _ = AutoModelForSequenceClassification.from_pretrained(
                model_name, revision=revision, force_download=True
            )
    except ValueError as e:
        if isinstance(e.__cause__, RepositoryNotFoundError):
            print("Repository not found:", e.__cause__)
        else:
            print("Download failed with:", e)


@app.command()
def refresh_model(
    model_name: str = typer.Option(
        ..., help="The Hugging Face model identifier to refresh."
    ),
    revision: str | None = typer.Option(
        None,
        help=(
            "Commit SHA (or branch/tag) to download instead of main. To serve it, pin the same value in "
            "ML_GATEWAY_MODEL_REVISIONS before starting the server."
        ),
    ),
):
    """
    Force-refresh a specific model from Hugging Face.
    """
    _refresh(model_name, revision)
    typer.echo("Model refresh complete.")


@app.command()
def refresh_all():
    """
    Force-refresh all preconfigured models (at their pinned revisions, if any).
    """
    typer.echo("Refreshing all preconfigured models…")

    for model_name in [*LABELS, *ATTACK_MODELS, *BIENCODER_MODELS]:
        _refresh(model_name)

    typer.echo("All models refreshed.")


BENCH_DESCRIPTIONS = [
    "Cross-site scripting in the admin panel of Foo CMS 2.1 allows remote attackers to inject script.",
    (
        "Zoho ManageEngine ServiceDesk Plus before 11306 is vulnerable to a privilege escalation issue where "
        "an authenticated low-privileged user can modify the request to access administrative functions. "
        "An attacker could exploit this to create new administrator accounts."
    ),
    (
        "A heap-based buffer overflow in the TIFF image parser of libexample before 1.4.2, when processing a "
        "crafted file with an oversized strip count, allows a remote attacker to execute arbitrary code or "
        "cause a denial of service (application crash) via a specially crafted image embedded in a document. "
        "The issue stems from a missing bounds check in tiff_read_strips() and affects all platforms; "
        "exploitation requires the victim to open the file but no further interaction."
    ),
]


@app.command()
def bench(
    model_name: str = typer.Option(
        "CIRCL/vulnerability-severity-classification-RoBERTa-base",
        help="Severity or attack-technique model to time.",
    ),
    iterations: int = typer.Option(60, min=3, help="Timed forward passes (after three warm-up passes)."),
    threads: int | None = typer.Option(None, min=1, help="torch intra-op threads; default: torch's choice."),
    mkldnn: bool = typer.Option(True, help="Use oneDNN kernels (--no-mkldnn tries torch's native ones)."),
) -> None:
    """
    Time single forward passes of one classifier on this host, outside gunicorn.

    Cycles through three built-in descriptions (short, typical, long) and
    prints latency percentiles and requests per second for the current
    thread count, so a slow host, a torch upgrade or a thread layout can be
    compared without any client in the loop.
    """
    import statistics
    import time

    import torch

    if threads is not None:
        torch.set_num_threads(threads)
    # The flag is a ContextProp descriptor in the stubs; setattr keeps mypy quiet.
    setattr(torch.backends.mkldnn, "enabled", mkldnn)
    classifier = get_model_instance(model_name) if model_name in LABELS else get_attack_model_instance(model_name)
    typer.echo(
        f"torch {torch.__version__}, {torch.get_num_threads()} threads, mkldnn={'on' if mkldnn else 'off'}, "
        f"CPU flags: {' '.join(sorted(cpu_flags())) or 'unknown'}"
    )
    typer.echo(f"{model_name} revision {classifier.revision}")
    for description in BENCH_DESCRIPTIONS:
        classifier.predict(description)
    latencies: list[float] = []
    for i in range(iterations):
        started = time.perf_counter()
        classifier.predict(BENCH_DESCRIPTIONS[i % len(BENCH_DESCRIPTIONS)])
        latencies.append(time.perf_counter() - started)
    latencies.sort()
    typer.echo(
        f"{iterations} passes: mean {statistics.mean(latencies) * 1000:.0f} ms, "
        f"p50 {latencies[len(latencies) // 2] * 1000:.0f} ms, "
        f"p90 {latencies[int(len(latencies) * 0.9)] * 1000:.0f} ms, "
        f"max {latencies[-1] * 1000:.0f} ms, {len(latencies) / sum(latencies):.2f} req/s"
    )


@app.command("check-quantization")
def check_quantization() -> None:
    """
    Quantize every classifier and run one forward pass, as the server would.

    Run this on the host before setting ML_GATEWAY_QUANTIZE=1: the int8
    kernels use vector instructions the CPU must support, and a process
    that lacks them dies with "Illegal instruction" (SIGILL). Better this
    command than every worker.
    """
    import time

    sample = "Zoho ManageEngine ServiceDesk Plus before 11306 allows unauthenticated remote code execution."
    typer.echo(f"Quantization engine: {select_engine()}; CPU flags: {' '.join(sorted(cpu_flags())) or 'unknown'}")
    for model_name in [*LABELS, *ATTACK_MODELS]:
        classifier = get_model_instance(model_name) if model_name in LABELS else get_attack_model_instance(model_name)
        started = time.perf_counter()
        classifier.predict(sample)
        fp32_ms = (time.perf_counter() - started) * 1000
        classifier.quantize()
        started = time.perf_counter()
        classifier.predict(sample)
        int8_ms = (time.perf_counter() - started) * 1000
        typer.echo(f"{model_name}: fp32 {fp32_ms:.0f} ms, int8 {int8_ms:.0f} ms, no illegal instruction.")
    typer.echo("Quantization works on this host.")


@app.command()
def backfill_index(
    dumps: list[Path] = typer.Option(
        ...,
        "--dumps",
        help="Vulnerability-Lookup NDJSON dump file or directory (repeatable; .gz accepted).",
    ),
    model: str = typer.Option(DEFAULT_BIENCODER_MODEL, help="Bi-encoder whose index to fill."),
    batch_size: int = typer.Option(64, min=1, help="Descriptions embedded per model call."),
    skip_existing: bool = typer.Option(
        False, help="Leave IDs already in the index alone (resume an interrupted run)."
    ),
    limit: int | None = typer.Option(None, min=1, help="Stop after this many indexed records."),
) -> None:
    """
    Embed every description in the dumps into the bi-encoder index.

    Safe to run while the server is up. One ID is indexed at most once per
    run (first occurrence wins; directories are read in sorted order, so
    cvelistv5 precedes fkie_nvd and nvd). The index is read from
    $ML_GATEWAY_INDEX_DIR (default ./index), like the server.
    """
    encoder = get_biencoder_instance(model)
    store = get_store(encoder)
    typer.echo(f"Index: {store.directory} ({store.count} IDs, revision {encoder.revision})")

    def progress(report: BackfillReport) -> None:
        typer.echo(f"{report.records} records read, {report.indexed} indexed…")

    report = backfill(
        encoder,
        store,
        dumps,
        batch_size=batch_size,
        skip_existing=skip_existing,
        limit=limit,
        progress=progress,
    )
    typer.echo(report.summary())
    typer.echo(f"Index now holds {store.count} IDs.")


@app.command("embed-dumps")
def embed_dumps_command(
    dumps: list[Path] = typer.Option(
        ...,
        "--dumps",
        help="Vulnerability-Lookup NDJSON dump file or directory (repeatable; .gz accepted).",
    ),
    output: Path = typer.Option(..., "--output", dir_okay=False, help="The .npz to write."),
    model: str = typer.Option(DEFAULT_BIENCODER_MODEL, help="Bi-encoder to embed with."),
    device: str = typer.Option("cuda", help="torch device, e.g. 'cuda', 'cuda:1' or 'cpu'."),
    batch_size: int = typer.Option(256, min=1, help="Descriptions embedded per model call."),
    limit: int | None = typer.Option(None, min=1, help="Stop after this many embedded records."),
) -> None:
    """
    Embed the dumps on this host (typically a GPU) into an .npz for import-index.

    Same extraction and deduplication as backfill-index, but no index is
    touched: the archive carries ids, float16 embeddings and the model
    revision, and is imported on the gateway with import-index.
    """
    if model not in BIENCODER_MODELS:
        typer.echo(f"Unknown model: {model}", err=True)
        raise typer.Exit(code=1)
    encoder = AttackBiEncoder(model, device=device)
    typer.echo(f"Embedding with {encoder.model_name} revision {encoder.revision} on {encoder.device}")

    def progress(report: BackfillReport) -> None:
        typer.echo(f"{report.records} records read, {report.indexed} embedded…")

    report = embed_dumps(encoder, dumps, output, batch_size=batch_size, limit=limit, progress=progress)
    typer.echo(report.summary().replace("indexed:", "embedded:"))
    typer.echo(f"Wrote {report.indexed} vectors to {output}.")


@app.command()
def import_index(
    file: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="An .npz with ids, embeddings and model_revision."
    ),
    model: str = typer.Option(DEFAULT_BIENCODER_MODEL, help="Bi-encoder whose index to fill."),
) -> None:
    """
    Import vectors computed elsewhere (e.g. on a GPU host) into the index.

    The archive's model_revision must equal the served model's revision.
    The index is read from $ML_GATEWAY_INDEX_DIR (default ./index), like
    the server.
    """
    encoder = get_biencoder_instance(model)
    store = get_store(encoder)
    try:
        imported = import_vectors(store, file, encoder.model_name, encoder.revision)
    except ValueError as e:
        typer.echo(f"Import refused: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Imported {imported} vectors; index now holds {store.count} IDs.")


if __name__ == "__main__":
    app()
