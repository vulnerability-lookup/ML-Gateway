from transformers import pipeline
from transformers.pipelines.base import Pipeline

"""
This module defines a model wrapper for the severity classification.
We use transformers.pipeline for text classification, which returns a list of label/score dictionaries when called

For example, pipeline("sentiment-analysis", model="...") and pipeline(text) might return [{ 'label': 'POSITIVE', 'score': 0.999 }]
"""

class SeverityModel:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.pipeline: Pipeline | None = None

    def load(self):
        """
        Load the Transformers pipeline for classification.
        This should be called once at startup.
        """
        # Initialize the text-classification pipeline with the given model
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name
        )

    def predict(self, text: str):
        """
        Run the model on the provided text.
        Returns a list of dicts: [{'label': ..., 'score': ...}].
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded. Call load() first.")
        return self.pipeline(text)

# Instantiate the SeverityModel with the desired pre-trained model
severity_model = SeverityModel(
    model_name="CIRCL/vulnerability-severity-classification-distilbert-base-uncased"
)
