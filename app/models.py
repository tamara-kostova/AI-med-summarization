
from typing import Dict, Optional

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """
    Detailed request model for text summarization with comprehensive documentation
    """
    text: str = Field(
        ..., 
        min_length=10,
        max_length=10000,
        description="Input medical text to be summarized. Must be between 10 and 10,000 characters.",
        example="A comprehensive medical case report detailing a rare neurological condition observed in a 45-year-old patient..."
    )
    summary_type: Optional[str] = Field(
        default="abstractive", 
        description="Type of summary generation method",
        pattern="^(abstractive|extractive)$",
        examples=["abstractive", "extractive"]
    )
    max_length: Optional[int] = Field(
        default=150, 
        ge=50, 
        le=500, 
        description="Maximum length of the generated summary in tokens/words",
        example=200
    )
    language: Optional[str] = Field(
        default="english",
        description="Language of the input text",
        example="english"
    )

class SummaryResponse(BaseModel):
    """
    Structured response model for summarization results
    """
    summary: str = Field(
        ..., 
        description="Generated summary of the input text",
        example="Key findings of the medical case including primary symptoms and treatment approach."
    )
    summary_type: str = Field(
        ..., 
        description="Type of summarization method used",
        example="abstractive"
    )
    original_length: int = Field(
        ..., 
        description="Number of characters in the original text",
        example=1500
    )
    summary_length: int = Field(
        ..., 
        description="Number of characters in the generated summary",
        example=250
    )
    language: Optional[str] = Field(
        default="english", 
        description="Language of the summary",
        example="english"
    )


class SummaryComparisonResponse(BaseModel):
    """
    Detailed comparison of summarization methods
    """
    extractive_rouge: Dict[str, float] = Field(
        ..., 
        description="ROUGE scores for extractive summarization",
        example={
            "rouge1": 0.45, 
            "rouge2": 0.23, 
            "rougeL": 0.38
        }
    )
    abstractive_rouge: Dict[str, float] = Field(
        ..., 
        description="ROUGE scores for abstractive summarization",
        example={
            "rouge1": 0.52, 
            "rouge2": 0.31, 
            "rougeL": 0.45
        }
    )

class SummaryComparisonRequest(TextRequest):
    """
    Request model for summary comparison
    """
    reference_summary: str = Field(
        ..., 
        min_length=10,
        max_length=1000,
        description="Ground truth summary for comparison"
    )