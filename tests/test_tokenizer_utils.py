from api.models.tokenizer_utils import clamp_tokenizer_max_length

"""
Unit tests for the tokenizer length clamp.

Uses lightweight stubs instead of real Hugging Face objects: the helper
only reads/writes plain attributes, and loading a real tokenizer would
require downloading a model. The sentinel below is the exact
``VERY_LARGE_INTEGER`` transformers reports when a repo's
``tokenizer_config.json`` carries no ``model_max_length`` — the
misconfiguration that made ``truncation=True`` a no-op and crashed the
Chinese MacBERT model's position embeddings on inputs over 512 tokens.
"""

UNSET_SENTINEL = 1000000000000000019884624838656


class StubTokenizer:
    def __init__(self, model_max_length: int) -> None:
        self.model_max_length = model_max_length


class StubConfig:
    def __init__(self, model_type: str, max_position_embeddings: int | None) -> None:
        self.model_type = model_type
        if max_position_embeddings is not None:
            self.max_position_embeddings = max_position_embeddings


def clamp(tokenizer: StubTokenizer, config: StubConfig) -> None:
    # The helper is typed against the transformers base classes; the stubs
    # are structurally compatible.
    clamp_tokenizer_max_length(tokenizer, config)  # type: ignore[arg-type]


def test_unset_limit_is_clamped_to_position_embeddings() -> None:
    tokenizer = StubTokenizer(UNSET_SENTINEL)
    clamp(tokenizer, StubConfig("bert", 512))
    assert tokenizer.model_max_length == 512


def test_roberta_reserves_two_offset_positions() -> None:
    tokenizer = StubTokenizer(UNSET_SENTINEL)
    clamp(tokenizer, StubConfig("roberta", 514))
    assert tokenizer.model_max_length == 512


def test_sane_limit_is_left_alone() -> None:
    tokenizer = StubTokenizer(512)
    clamp(tokenizer, StubConfig("roberta", 514))
    assert tokenizer.model_max_length == 512


def test_config_without_position_limit_is_left_alone() -> None:
    tokenizer = StubTokenizer(UNSET_SENTINEL)
    clamp(tokenizer, StubConfig("some-encoder", None))
    assert tokenizer.model_max_length == UNSET_SENTINEL
