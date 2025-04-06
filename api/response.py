from typing import Dict, Optional

from pydantic import BaseModel, Field


class SummaryResponse(BaseModel):
    """
    Structured response model for summarization results
    """

    summary: str = Field(
        ...,
        description="Generated summary of the input text",
        example="Key findings of the case including primary symptoms and treatment approach.",
    )
    summary_type: str = Field(
        ..., description="Type of summarization method used", example="abstractive"
    )
    original_length: int = Field(
        ..., description="Number of characters in the original text", example=1500
    )
    summary_length: int = Field(
        ..., description="Number of characters in the generated summary", example=250
    )
    language: Optional[str] = Field(
        default="english", description="Language of the summary", example="english"
    )
    filename: Optional[str] = Field(
        default=None,
        description="Name of the uploaded file (for file uploads only)",
        example="patient_report.pdf",
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="Time taken to generate the summary in seconds",
        example=1.25,
    )


class SummaryComparisonResponse(BaseModel):
    """
    Detailed comparison of summarization methods
    """

    extractive_rouge: Dict[str, float] = Field(
        ...,
        description="ROUGE scores for extractive summarization",
        example={"rouge1": 0.45, "rouge2": 0.23, "rougeL": 0.38},
    )
    abstractive_rouge: Dict[str, float] = Field(
        ...,
        description="ROUGE scores for abstractive summarization",
        example={"rouge1": 0.52, "rouge2": 0.31, "rougeL": 0.45},
    )
    recommended_method: str = Field(
        ...,
        description="Recommended summarization method based on ROUGE scores",
        example="abstractive",
    )
