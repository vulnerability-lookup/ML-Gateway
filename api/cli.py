import typer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    _ = AutoTokenizer.from_pretrained(model_name, force_download=True)
    _ = AutoModelForSequenceClassification.from_pretrained(
        model_name, force_download=True
    )
    typer.echo("Model refresh complete.")


@app.command()
def refresh_all():
    """
    Force-refresh all preconfigured models.
    """
    typer.echo("Refreshing all preconfigured models...")

    models = [
        "CIRCL/vulnerability-severity-classification-RoBERTa-base",
        "CIRCL/vulnerability-severity-classification-distilbert-base-uncased",
    ]

    for model_name in models:
        typer.echo(f"Refreshing model: {model_name}")
        _ = AutoTokenizer.from_pretrained(model_name, force_download=True)
        _ = AutoModelForSequenceClassification.from_pretrained(
            model_name, force_download=True
        )

    typer.echo("All models refreshed.")


if __name__ == "__main__":
    app()
