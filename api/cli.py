import typer
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from api.models.attack_model import ATTACK_MODELS
from api.models.biencoder_model import BIENCODER_MODELS
from api.models.severity_model import LABELS

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


if __name__ == "__main__":
    app()
