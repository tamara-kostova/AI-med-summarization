from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models import SummaryComparisonResponse, SummaryResponse, TextRequest, SummaryComparisonRequest
from summarization.abstractive import generate_abstractive_summary
from summarization.extractive import generate_extractive_summary
from summarization.evaluate import compare_summaries

router = APIRouter(tags=["Summarization"])

@router.post(
    "/summarize/", 
    response_model=SummaryResponse,
    summary="Generate a summary of medical text",
    description="Generates a summary using either abstractive or extractive summarization methods."
)
async def summarize_text(
    request: TextRequest
):
    """
    Generate a summary based on the specified type with comprehensive input validation
    
    - **text**: Input medical text to summarize
    - **summary_type**: Choose between 'abstractive' or 'extractive' summarization
    - **max_length**: Control the maximum length of the summary
    """
    try:
        # Validate text length and content
        if len(request.text.strip()) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Text is too short. Minimum 10 characters required."
            )
        
        # Choose summarization method
        if request.summary_type == "abstractive":
            summary = generate_abstractive_summary(
                request.text, 
                max_length=request.max_length
            )
        else:
            summary = generate_extractive_summary(
                request.text, 
                num_sentences=int(request.max_length/50)
            )
        
        return SummaryResponse(
            summary=summary,
            summary_type=request.summary_type,
            original_length=len(request.text),
            summary_length=len(summary),
            language=request.language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/compare-summaries/", 
    response_model=SummaryComparisonResponse,
    summary="Compare summarization methods",
    description="Evaluates and compares extractive and abstractive summarization methods using ROUGE scores."
)
async def evaluate_summaries(
    request: SummaryComparisonRequest
):
    """
    Compare different summarization methods by evaluating ROUGE scores
    
    - **request**: Input text to summarize
    - **reference_summary**: Known good summary for comparison
    """
    try:
        comparison_results = compare_summaries(request.text, request.reference_summary)
        return comparison_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))