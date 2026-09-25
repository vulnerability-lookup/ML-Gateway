import os
import warnings

import torch
from torch.ao.quantization import quantize_dynamic

"""
Optional int8 quantization of the classification models.

``ML_GATEWAY_QUANTIZE=1`` replaces every ``torch.nn.Linear`` of the severity
and attack-technique classifiers with a dynamically quantized one: weights
are stored as int8 and activations quantized on the fly per call. On CPU
that roughly halves the cost of a forward pass for the short descriptions
the gateway sees, at a small accuracy cost (see the README for the
measured figures). The bi-encoder is never quantized: its embeddings must
reproduce the training-time contract exactly.

Quantization runs in each worker after the fork (from the lifespan), not in
the preloading master: converting the weights runs torch kernels, and any
torch work in the master before the fork can leave the workers with a
broken OpenMP thread pool. The fp32 pages inherited from the master are
never written to, so they stay shared; each worker adds only its int8
copy, about 160 MB per RoBERTa-base model.
"""

QUANTIZE_ENV = "ML_GATEWAY_QUANTIZE"


def quantization_enabled() -> bool:
    return os.environ.get(QUANTIZE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def quantize_linear_layers(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize every ``Linear`` of ``model`` to dynamic int8, in place."""
    with warnings.catch_warnings():
        # torch.ao.quantization is deprecated in favour of torchao, which
        # is not a dependency; the eager API still ships and works.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", UserWarning)
        quantized: torch.nn.Module = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    return quantized
