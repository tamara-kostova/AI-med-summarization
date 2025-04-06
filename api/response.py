from typing import Dict, Optional

from pydantic import BaseModel, Field, validator


class TextSummaryRequest(BaseModel):
    text: str
    summary_type: str = "abstractive"
    max_length: int = 150


class TextRequest(BaseModel):
    """
    Detailed request model for text summarization with comprehensive documentation
    """

    text: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="Input text to be summarized. Must be between 10 and 50,000 characters.",
        example="A comprehensive case report detailing a rare condition observed in a 45-year-old patient...",
    )
    summary_type: Optional[str] = Field(
        default="abstractive",
        description="Type of summary generation method",
        examples=["abstractive", "extractive"],
    )
    max_length: Optional[int] = Field(
        default=150,
        ge=50,
        le=1000,
        description="Maximum length of the generated summary in tokens/words",
        example=200,
    )
    language: Optional[str] = Field(
        default="english", description="Language of the input text", example="english"
    )

    @validator("summary_type")
    def validate_summary_type(cls, v):
        if v not in ["abstractive", "extractive"]:
            raise ValueError(
                "Summary type must be either 'abstractive' or 'extractive'"
            )
        return v


class FileUploadRequest(BaseModel):
    """
    Request model for file upload summarization
    """

    summary_type: str = Field(
        default="abstractive",
        description="Type of summarization to perform",
        examples=["abstractive", "extractive"],
    )
    max_length: int = Field(
        default=150,
        ge=50,
        le=1000,
        description="Maximum length of the generated summary",
        example=200,
    )

    @validator("summary_type")
    def validate_summary_type(cls, v):
        if v not in ["abstractive", "extractive"]:
            raise ValueError(
                "Summary type must be either 'abstractive' or 'extractive'"
            )
        return v


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


class SummaryComparisonRequest(TextRequest):
    """
    Request model for summary comparison
    """

    reference_summary: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Ground truth summary for comparison",
        example="The patient presented with symptoms of...",
    )
