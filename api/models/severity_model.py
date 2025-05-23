import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
This module defines a model wrapper for the severity classification.
We use transformers.pipeline for text classification, which returns a list of label/score dictionaries when called

For example, pipeline("sentiment-analysis", model="...") and pipeline(text) might return [{ 'label': 'POSITIVE', 'score': 0.999 }]
"""

# Constants
MODEL_NAME = "CIRCL/vulnerability-severity-classification-distilbert-base-uncased"
LABELS = ["Low", "Medium", "High", "Critical"]


class SeverityClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
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
        confidence = probabilities[0][predicted_index].item()
        label = LABELS[predicted_index]

        return {"severity": label, "confidence": round(confidence, 4)}


# Singleton instance of the model
severity_model = SeverityClassifier()
