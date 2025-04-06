from typing import Optional

from pydantic import BaseModel, Field, validator


class TextSummaryRequest(BaseModel):
    text: str
    summary_type: str = "abstractive"
    model: str = "t5-small"
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
    extractive_model: str = "bert"
    abstractive_model: str = "t5-small"
