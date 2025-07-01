import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
This module defines a model wrapper for the severity classification.
"""


class SeverityClassifier:
    def __init__(self, model_name: str, labels: list[str]):
        self.model_name = model_name
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Disable dropout etc.

    def predict(self, description: str):
        inputs = self.tokenizer(
            description, return_tensors="pt", truncation=True, padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get top class index and confidence
        predicted_index = torch.argmax(probabilities, dim=-1).item()
        return {
            "severity": self.labels[predicted_index],
            "confidence": round(probabilities[0][predicted_index].item(), 4),
        }


# Dictionary of known models
_model_cache: dict[str, SeverityClassifier] = {}

# Label sets can vary by language or domain
LABELS = {
    "CIRCL/vulnerability-severity-classification-distilbert-base-uncased": [
        "Low",
        "Medium",
        "High",
        "Critical",
    ],
    "CIRCL/vulnerability-severity-classification-RoBERTa-base": [
        "Low",
        "Medium",
        "High",
        "Critical",
    ],
    "CIRCL/vulnerability-severity-classification-chinese-macbert-base": [
        "低",
        "中",
        "高",
    ],
}


def get_model_instance(model_name: str) -> SeverityClassifier:
    if model_name not in _model_cache:
        labels = LABELS.get(model_name)
        if not labels:
            raise ValueError(f"Unknown model: {model_name}")
        _model_cache[model_name] = SeverityClassifier(model_name, labels)
    return _model_cache[model_name]
