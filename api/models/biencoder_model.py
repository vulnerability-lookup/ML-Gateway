import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer

from api.models.tokenizer_utils import clamp_tokenizer_max_length

"""
This module wraps the ATT&CK bi-encoder used for retrieval.

Unlike the classification head, the bi-encoder is a plain encoder with no
task-specific layer: every text — vulnerability description or technique
text — becomes one L2-normalized vector and similarity is a cosine. The
scoring contract (mean pooling over the attention mask, per-side truncation
lengths, and the affine ``logit_scale`` / ``logit_bias`` that turns a cosine
into the training-time probability) is documented in VulnTrain's
``attack-biencoder-retrieval`` page and must be reproduced exactly here.
"""

# ``"Name. Description"`` per active enterprise ATT&CK technique, built from
# the MITRE STIX bundle the same way VulnTrain builds the model's shipped
# ``technique_texts.json`` (citations, markdown links and HTML tags
# stripped; revoked and deprecated objects skipped). Only used for
# techniques outside the model's trained vocabulary — the trained ones use
# the texts shipped with the model so the numbers match the validator.
_TECHNIQUE_TEXTS_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "attack_technique_texts.json"
)


@lru_cache(maxsize=1)
def technique_texts() -> dict[str, str]:
    with open(_TECHNIQUE_TEXTS_FILE, encoding="utf-8") as f:
        texts: dict[str, str] = json.load(f)
    return texts


class AttackBiEncoder:
    """Wraps a Hugging Face bi-encoder for ATT&CK retrieval.

    Attributes:
        model_name: Hugging Face repository identifier (e.g.
            ``CIRCL/vulnerability-attack-technique-biencoder``).
        revision: Commit SHA of the loaded snapshot, or ``None`` for sources
            without revision metadata (e.g. a local path). Vectors are only
            comparable within one revision, so the index stores it too.
        labels: Technique IDs the model was trained on, in training order.
        logit_scale, logit_bias: Affine constants mapping a cosine to the
            training-time logit; ``sigmoid(scale * cos + bias)`` is the
            CVE -> technique probability.
        dimension: Size of the embedding vectors.
    """

    def __init__(self, model_name: str, device: str | None = None):
        """``device`` is a torch device string (``"cuda"``, ``"cpu"``); the
        server keeps the default CPU, the bulk embedding CLI may pass a GPU."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device is not None:
            self.model.to(device)
        self.device = next(self.model.parameters()).device
        self.model.eval()  # Disable dropout etc.
        clamp_tokenizer_max_length(self.tokenizer, self.model.config)
        # See SeverityClassifier for the caveat about ``_commit_hash`` being
        # private transformers API.
        self.revision: str | None = getattr(self.model.config, "_commit_hash", None)

        settings = getattr(self.model.config, "biencoder", None)
        if not isinstance(settings, dict):
            raise ValueError(
                f"{model_name} carries no 'biencoder' block in its config.json; "
                "it is not a VulnTrain bi-encoder release."
            )
        self.labels: list[str] = list(settings["labels"])
        self.logit_scale = float(settings["logit_scale"])
        self.logit_bias = float(settings["logit_bias"])
        # Truncation lengths are part of the scoring contract: technique
        # texts were encoded at ``technique_max_length`` during training,
        # vulnerability descriptions at the tokenizer's full limit.
        self.technique_max_length = int(settings["technique_max_length"])
        self.vulnerability_max_length = int(self.tokenizer.model_max_length)
        self.dimension = int(self.model.config.hidden_size)

        self.trained_technique_texts = self._load_trained_technique_texts()
        self._technique_vectors: dict[str, NDArray[np.float32]] = {}
        vectors = self.embed_techniques(
            [self.trained_technique_texts[technique] for technique in self.labels]
        )
        for technique, vector in zip(self.labels, vectors):
            self._technique_vectors[technique] = vector

    def _load_trained_technique_texts(self) -> dict[str, str]:
        """Read the ``technique_texts.json`` shipped next to the weights."""
        local_dir = Path(self.model_name)
        if local_dir.is_dir():
            path = local_dir / "technique_texts.json"
        else:
            # Pin the same snapshot the weights came from; with
            # ``HF_HUB_OFFLINE=1`` this resolves from the local cache.
            path = Path(
                hf_hub_download(
                    self.model_name, "technique_texts.json", revision=self.revision
                )
            )
        with open(path, encoding="utf-8") as f:
            texts: dict[str, str] = json.load(f)
        missing = [technique for technique in self.labels if technique not in texts]
        if missing:
            raise ValueError(
                f"{self.model_name}: technique_texts.json lacks trained techniques {missing}"
            )
        return texts

    def embed(
        self, texts: list[str], max_length: int, batch_size: int = 64
    ) -> NDArray[np.float32]:
        """Mean-pooled, L2-normalized embeddings, shape ``(len(texts), dimension)``.

        Pooling uses the attention mask (not the ``<s>`` vector) and the
        result is normalized before any dot product, as at training time.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
        return torch.cat(chunks).numpy().astype(np.float32)

    def embed_vulnerabilities(self, descriptions: list[str]) -> NDArray[np.float32]:
        return self.embed(descriptions, self.vulnerability_max_length)

    def embed_techniques(self, texts: list[str]) -> NDArray[np.float32]:
        return self.embed(texts, self.technique_max_length)

    def technique_vector(
        self, technique_id: str
    ) -> tuple[NDArray[np.float32], bool] | None:
        """Return ``(vector, in_vocabulary)`` for a technique, or ``None``.

        Trained techniques come pre-computed from the shipped texts. Any
        other active enterprise technique is embedded on first use from the
        bundled STIX-derived text table and flagged as out-of-vocabulary:
        the paper measures label-holdout recall@5 at 0.12 for those, so
        interfaces should present them as such.
        """
        vector = self._technique_vectors.get(technique_id)
        if vector is not None:
            return vector, technique_id in self.trained_technique_texts
        text = technique_texts().get(technique_id)
        if text is None:
            return None
        vector = self.embed_techniques([text])[0]
        self._technique_vectors[technique_id] = vector
        return vector, False

    def probability(self, cosine: float) -> float:
        """Training-time CVE -> technique probability for a cosine."""
        logit = self.logit_scale * cosine + self.logit_bias
        return float(1.0 / (1.0 + np.exp(-logit)))


# Bi-encoder releases this endpoint may load. Like ``ATTACK_MODELS`` this is
# an allow-list keeping the public endpoints from loading arbitrary Hub
# repositories; everything model-specific comes from the release's config.
BIENCODER_MODELS = [
    "CIRCL/vulnerability-attack-technique-biencoder",
]

_model_cache: dict[str, AttackBiEncoder] = {}


def get_biencoder_instance(model_name: str) -> AttackBiEncoder:
    if model_name not in _model_cache:
        if model_name not in BIENCODER_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        _model_cache[model_name] = AttackBiEncoder(model_name)
    return _model_cache[model_name]


def preload_models() -> None:
    """Load every model in ``BIENCODER_MODELS`` into the in-memory cache.

    Called at import time so that running gunicorn with ``--preload``
    populates the cache (weights and the trained technique vectors) in the
    master process; forked workers then share them via copy-on-write.
    """
    for model_name in BIENCODER_MODELS:
        get_biencoder_instance(model_name)
