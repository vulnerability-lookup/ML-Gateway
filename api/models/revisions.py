import os

"""
Pin the Hugging Face revision a model is loaded from.

``ML_GATEWAY_MODEL_REVISIONS`` lists ``<model>=<commit sha>`` pairs, separated
by commas (or whitespace), for example::

    ML_GATEWAY_MODEL_REVISIONS="CIRCL/vulnerability-severity-classification-RoBERTa-base=987d2c3a"

A pinned model is loaded from that snapshot instead of the Hub's ``main``,
which is how an operator reverts to a previous revision, or keeps one
while a new one is evaluated. ``ml-gw-cli refresh-model`` /
``refresh-all`` download the pinned revision, so with ``HF_HUB_OFFLINE=1``
the server finds it in the cache. Models not listed follow ``main`` as
before. Every response still reports the revision actually loaded.
"""

MODEL_REVISIONS_ENV = "ML_GATEWAY_MODEL_REVISIONS"


def pinned_revisions() -> dict[str, str]:
    raw = os.environ.get(MODEL_REVISIONS_ENV, "")
    pins: dict[str, str] = {}
    for entry in raw.replace(",", " ").split():
        model, separator, revision = entry.partition("=")
        if not separator or not model or not revision:
            raise ValueError(f"{MODEL_REVISIONS_ENV}: expected <model>=<revision>, got {entry!r}")
        pins[model.strip()] = revision.strip()
    return pins


def pinned_revision(model_name: str) -> str | None:
    """The revision ``model_name`` is pinned to, or ``None`` for the Hub's ``main``."""
    return pinned_revisions().get(model_name)
