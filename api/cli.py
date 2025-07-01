import typer
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = typer.Typer(help="Utility CLI for managing NLP models.")


@app.command()
def refresh_model(
    model_name: str = typer.Option(
        ..., help="The Hugging Face model identifier to refresh."
    )
):
    """
    Force-refresh a specific model from Hugging Face.
    """
    typer.echo(f"Refreshing model: {model_name}")
    try:
        _ = AutoTokenizer.from_pretrained(model_name, force_download=True)
        _ = AutoModelForSequenceClassification.from_pretrained(
            model_name, force_download=True
        )
        typer.echo("Model refresh complete.")
    except ValueError as e:
        if isinstance(e.__cause__, RepositoryNotFoundError):
            print("Repository not found:", e.__cause__)
        else:
            print("Download failed with:", e)


@app.command()
def refresh_all():
    """
    Force-refresh all preconfigured models.
    """
    typer.echo("Refreshing all preconfigured models…")

    models = [
        "CIRCL/vulnerability-severity-classification-RoBERTa-base",
        "CIRCL/vulnerability-severity-classification-distilbert-base-uncased",
        "CIRCL/vulnerability-severity-classification-chinese-macbert-base",
    ]

    for model_name in models:
        typer.echo(f"Refreshing model: {model_name}")
        try:
            _ = AutoTokenizer.from_pretrained(model_name, force_download=True)
            _ = AutoModelForSequenceClassification.from_pretrained(
                model_name, force_download=True
            )
        except ValueError as e:
            if isinstance(e.__cause__, RepositoryNotFoundError):
                print("Repository not found:", e.__cause__)
            else:
                print("Download failed with:", e)

    typer.echo("All models refreshed.")


if __name__ == "__main__":
    app()
