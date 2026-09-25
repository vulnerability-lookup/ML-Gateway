import pytest
from typer.testing import CliRunner

from api import cli
from api.models.revisions import MODEL_REVISIONS_ENV, pinned_revision, pinned_revisions

"""
Tests for the model revision pin and the CLI option that downloads one.
"""

MODEL = "CIRCL/vulnerability-severity-classification-RoBERTa-base"
SHA = "987d2c3a2d521db0cda327e1bb77248c381f057c"


def test_unset_means_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_REVISIONS_ENV, raising=False)
    assert pinned_revisions() == {}
    assert pinned_revision(MODEL) is None


def test_pins_are_parsed_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_REVISIONS_ENV, f" {MODEL}={SHA},\n CIRCL/other=v1.2 ")
    assert pinned_revisions() == {MODEL: SHA, "CIRCL/other": "v1.2"}
    assert pinned_revision(MODEL) == SHA
    assert pinned_revision("CIRCL/unlisted") is None


@pytest.mark.parametrize("value", ["just-a-sha", f"{MODEL}=", f"={SHA}"])
def test_malformed_pins_are_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(MODEL_REVISIONS_ENV, value)
    with pytest.raises(ValueError, match=MODEL_REVISIONS_ENV):
        pinned_revisions()


def test_refresh_model_passes_the_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(cli, "_refresh", lambda model_name, revision=None: calls.append((model_name, revision)))
    result = CliRunner().invoke(cli.app, ["refresh-model", "--model-name", MODEL, "--revision", SHA])
    assert result.exit_code == 0, result.output
    assert calls == [(MODEL, SHA)]
    result = CliRunner().invoke(cli.app, ["refresh-model", "--model-name", MODEL])
    assert result.exit_code == 0, result.output
    assert calls[-1] == (MODEL, None)


def test_bench_reports_latency_without_a_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stub:
        revision = SHA

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, description: str) -> dict[str, str]:
            self.calls += 1
            return {"severity": "Low"}

    stub = Stub()
    monkeypatch.setattr(cli, "get_model_instance", lambda model_name: stub)
    result = CliRunner().invoke(cli.app, ["bench", "--iterations", "9", "--threads", "1"])
    assert result.exit_code == 0, result.output
    assert f"revision {SHA}" in result.output
    assert "9 passes: mean" in result.output and "req/s" in result.output
    assert stub.calls == 12  # three warm-up passes plus the timed ones
