import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.deps import get_evaluator, get_summarizer
from api.response import SummaryComparisonRequest, SummaryResponse, TextSummaryRequest
from summarization.evaluator import Evaluator
from summarization.summarizer import Summarizer
from summarization.utils import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Summarization"])


templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post(
    "/summarize-pdf",
    response_model=SummaryResponse,
    summary="Generate a summary from an uploaded PDF file",
    description="Extracts text from a PDF file and generates a summary using either abstractive or extractive methods.",
)
async def summarize_pdf(
    summarizer: Summarizer = Depends(get_summarizer),
    file: UploadFile = File(...),
    summary_type: str = Form(default="abstractive"),
):
    """
    Generate a summary from PDF content

    - **file**: PDF file to be summarized
    - **summary_type**: 'abstractive' or 'extractive' summarization method
    - **max_length**: Maximum length of the summary
    """
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        summary = summarizer.generate_pdf_summary(file_bytes, summary_type)

        text = extract_text_from_pdf(file_bytes)

        return SummaryResponse(
            summary=summary,
            summary_type=summary_type,
            original_length=len(text),
            summary_length=len(summary),
            filename=file.filename,
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post(
    "/summarize/",
    response_model=SummaryResponse,
    summary="Generate a summary of text",
    description="Generates a summary using either abstractive or extractive summarization methods.",
)
async def summarize(
    request_data: TextSummaryRequest, summarizer: Summarizer = Depends(get_summarizer)
):
    """
    Generate a summary based on the specified type with comprehensive input validation

    - **text**: Input text to summarize
    - **summary_type**: Choose between 'abstractive' or 'extractive' summarization
    - **max_length**: Control the maximum length of the summary
    """
    text = request_data.text
    summary_type = request_data.summary_type
    max_length = request_data.max_length
    logger.info(f"Summarizing text using {summary_type} method")

    if len(text.strip()) < 10:
        logger.warning("Text submission too short")
        raise HTTPException(
            status_code=400, detail="Text is too short. Minimum 10 characters required."
        )

    if summary_type not in ["abstractive", "extractive"]:
        logger.warning(f"Invalid summary type: {summary_type}")
        raise HTTPException(
            status_code=400,
            detail="Summary type must be either 'abstractive' or 'extractive'",
        )

    try:
        if summary_type == "abstractive":
            summary = summarizer.generate_abstractive_summary(text)
        else:
            summary = summarizer.generate_extractive_summary(text, max_length // 50)

        return SummaryResponse(
            summary=summary,
            summary_type=summary_type,
            original_length=len(text),
            summary_length=len(summary),
        )
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )


@router.post(
    "/compare-summaries/",
    summary="Compare different summarization methods",
    description="Compare extractive and abstractive summarization methods using ROUGE scores.",
)
async def evaluate_summaries(
    request: SummaryComparisonRequest, evaluator: Evaluator = Depends(get_evaluator)
):
    """
    Compare extractive and abstractive summarization methods using ROUGE scores.
    """
    try:
        logger.info("Comparing summary methods")
        results = evaluator.compare_summaries(request.text, request.reference_summary)
        return results
    except Exception as e:
        logger.error(f"Summary comparison error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Summary comparison error: {str(e)}"
        )
