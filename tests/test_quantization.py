import pytest
import torch

from api.models.quantization import QUANTIZE_ENV, quantization_enabled, quantize_linear_layers

"""
Tests for the optional int8 quantization: the switch and the conversion
itself, on a tiny model so no Hugging Face weights are needed.
"""


@pytest.mark.parametrize(
    ("value", "enabled"),
    [("1", True), ("true", True), (" Yes ", True), ("on", True), ("0", False), ("", False), ("no", False)],
)
def test_switch_parses_common_spellings(monkeypatch: pytest.MonkeyPatch, value: str, enabled: bool) -> None:
    monkeypatch.setenv(QUANTIZE_ENV, value)
    assert quantization_enabled() is enabled


def test_switch_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(QUANTIZE_ENV, raising=False)
    assert quantization_enabled() is False


def test_linear_layers_become_int8_and_keep_their_outputs() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4)).eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        before = model(x)
    quantized = quantize_linear_layers(model)
    assert quantized is model  # converted in place
    assert all(type(layer).__name__ == "Linear" and "quantized" in type(layer).__module__
               for layer in quantized if isinstance(layer, torch.nn.Module) and not isinstance(layer, torch.nn.ReLU))
    # Quantized modules expose the packed weight through a method.
    weight = getattr(quantized[0], "weight")
    assert weight().dtype == torch.qint8
    with torch.no_grad():
        after = quantized(x)
    assert torch.allclose(before, after, atol=0.05)
