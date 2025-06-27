from api.models.severity_model import get_model_instance
from api.schemas import SeverityRequest

"""
The service layer contains the business logic for classification.
It receives the input text, selects the appropriate model, and formats the output.
"""


def classify_severity(request: SeverityRequest):
    """
    Classify the severity of a vulnerability description.
    Returns a dict with 'severity' and 'confidence'.
    """
    try:
        model_instance = get_model_instance(request.model)
    except ValueError as e:
        return {
            "severity": None,
            "confidence": 0.0,
            "error": str(e),
        }

    output = model_instance.predict(request.description)
    if output:
        # Extract label and score from the model output
        label = output.get("severity")
        score = output.get("confidence", 0.0)
    else:
        label, score = None, 0.0
    return {"severity": label, "confidence": float(score)}
