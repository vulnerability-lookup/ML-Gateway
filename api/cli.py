from pathlib import Path

import typer
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from api.backfill import BackfillReport, backfill, embed_dumps, import_vectors
from api.models.attack_model import ATTACK_MODELS
from api.models.biencoder_model import BIENCODER_MODELS, AttackBiEncoder, get_biencoder_instance
from api.models.severity_model import LABELS
from api.schemas import DEFAULT_BIENCODER_MODEL
from api.services.retrieval_service import get_store

app = typer.Typer(help="Utility CLI for managing NLP models.")


def _refresh(model_name: str) -> None:
    """Force-download one model's tokenizer, weights and companion files."""
    typer.echo(f"Refreshing model: {model_name}")
    try:
        _ = AutoTokenizer.from_pretrained(model_name, force_download=True)
        if model_name in BIENCODER_MODELS:
            # A plain encoder, shipped with the technique texts it was
            # trained against; the server reads both from the cache.
            _ = AutoModel.from_pretrained(model_name, force_download=True)
            _ = hf_hub_download(model_name, "technique_texts.json", force_download=True)
        else:
            _ = AutoModelForSequenceClassification.from_pretrained(
                model_name, force_download=True
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
    )
):
    """
    Force-refresh a specific model from Hugging Face.
    """
    _refresh(model_name)
    typer.echo("Model refresh complete.")


@app.command()
def refresh_all():
    """
    Force-refresh all preconfigured models.
    """
    typer.echo("Refreshing all preconfigured models…")

    for model_name in [*LABELS, *ATTACK_MODELS, *BIENCODER_MODELS]:
        _refresh(model_name)

    typer.echo("All models refreshed.")


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
