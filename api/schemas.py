from pydantic import BaseModel, Field


class SeverityRequest(BaseModel):
    description: str
    model: str = Field(
        default="CIRCL/vulnerability-severity-classification-RoBERTa-base",
        description="Hugging Face model identifier to use for classification",
    )
