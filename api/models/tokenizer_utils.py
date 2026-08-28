from transformers import PretrainedConfig, PreTrainedTokenizerBase

"""
Shared tokenizer helpers for the model wrappers.
"""


def clamp_tokenizer_max_length(
    tokenizer: PreTrainedTokenizerBase, config: PretrainedConfig
) -> None:
    """Ensure the tokenizer's truncation limit fits the model's positions.

    Some fine-tuned repos are uploaded without ``model_max_length`` in
    their ``tokenizer_config.json``; transformers then reports a huge
    sentinel value (``VERY_LARGE_INTEGER``) and ``truncation=True``
    never actually truncates, so a long input overflows the model's
    position-embedding table at inference time. When the tokenizer's
    limit exceeds ``max_position_embeddings``, clamp it to the number of
    positions the model can hold. RoBERTa-style models reserve the first
    two positions for the padding offset, so two fewer tokens fit than
    the table size suggests.
    """
    max_positions = getattr(config, "max_position_embeddings", None)
    if max_positions is None or tokenizer.model_max_length <= int(max_positions):
        return
    offset = 2 if getattr(config, "model_type", "") in ("roberta", "xlm-roberta") else 0
    tokenizer.model_max_length = int(max_positions) - offset
