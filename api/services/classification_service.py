from api.models.severity_model import severity_model

"""
The service layer contains the business logic for classification.
It receives the input text, calls the model, and formats the output
"""

def classify_severity(description: str):
    """
    Classify the severity of a vulnerability description.
    Returns a dict with 'severity' and 'confidence'.
    """
    outputs = severity_model.predict(description)
    # Pipeline returns a list; take the first result
    if outputs:
        result = outputs[0]
        # Extract label and score from the model output
        label = result.get("label")
        score = result.get("score", 0.0)
    else:
        label, score = None, 0.0
    return {"severity": label, "confidence": float(score)}
