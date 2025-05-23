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
    output = severity_model.predict(description)
    # Pipeline returns a list; take the first result
    if output:
        # Extract label and score from the model output
        label = output.get("severity")
        score = output.get("confidence", 0.0)
    else:
        label, score = None, 0.0
    return {"severity": label, "confidence": float(score)}
