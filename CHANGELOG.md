# Changelog


## Release 0.4.0 (2025-06-27)

Multilingual model support for severity classification.

- Added support for specifying a Hugging Face model via a new model field in the SeverityRequest payload.
- Default model remains CIRCL/vulnerability-severity-classification-RoBERTa-base.
- New models (e.g., Chinese-language severity classifiers) can now be selected dynamically at request time.
- Introduced a get_model_instance() registry to load and cache models on demand.
- Preserved clean architecture with model selection handled in the service layer.


## Release 0.3.0 (2025-05-26)

Added a cli with two commands in order to refresh the models from Hugging Face.
A command to force-refresh a specific model and a command to force-refresh all preconfigured models.


## Release 0.2.0 (2025-05-23)

Refactored ``severity_model.py`` to use a manual inference in order
to have full control over performance, consistency, and debugging.
pytorch is directly used instead of the transformers Pipeline. The Pipeline
was using internal heuristics to map logits to labels (via softmax + sorting)
and other approximations (rounding, label normalization, etc.).


## Release 0.1.0 (2025-05-22)

First working release release.
It provides a service and router for a text classification model.
