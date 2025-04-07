import logging

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.abstractive.bart import BartSummarizer
from summarization.abstractive.t5 import T5AbstractiveSummarizer
from summarization.extractive.bert import BertSummarizer
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.extractive.textrank import TextRankerSummarizer
from summarization.utils import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self):
        self.extractive_models = {
            "textrank": TextRankerSummarizer(),
            "bert": BertSummarizer(),
        }
        self.abstractive_models = {
            "t5-small": T5AbstractiveSummarizer(
                extractive_summarizer=TextRankerSummarizer()
            ),
            "bart": BartSummarizer(),
        }
        print(f"Initialized summarizer")

    def generate_summary(
        self,
        text: str,
        summary_type: str = "abstractive",
        model_name: str = "t5-small",
        max_length: int = 150,
    ) -> str:
        """Generate summary from PDF file content based on specified summary type"""
        logger.info(f"Generating {summary_type} summary")

        if summary_type.lower() == "extractive":
            extractive_summarizer: ExtractiveSummarizer = self.extractive_models.get(
                model_name
            )
            return extractive_summarizer.generate_extractive_summary(text)
        abstractive_summarizer: AbstractiveSummarizer = self.abstractive_models.get(
            model_name
        )
        return abstractive_summarizer.generate_abstractive_summary(text=text)

    def generate_pdf_summary(
        self,
        file_bytes: bytes,
        summary_type: str = "abstractive",
        model_name: str = "t5-small",
        max_length: int = 150,
    ) -> str:
        """Generate summary from PDF file content based on specified summary type"""
        logger.info(f"Generating {summary_type} summary for PDF")

        text = extract_text_from_pdf(file_content=file_bytes)

        if not text or len(text.strip()) < 10:
            return "Insufficient text content found in the PDF for summarization."

        if summary_type.lower() == "extractive":
            extractive_summarizer: ExtractiveSummarizer = self.extractive_models.get(
                model_name
            )
            return extractive_summarizer.generate_extractive_summary(text)

        abstractive_summarizer: AbstractiveSummarizer = self.abstractive_models.get(
            model_name
        )
        return abstractive_summarizer.generate_abstractive_summary(text=text)
