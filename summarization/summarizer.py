import logging
from groq import Groq

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.abstractive.bart import BartSummarizer
from summarization.abstractive.deepseek import DeepSeekSummarizer
from summarization.abstractive.distilbart import DistilBARTSummarizer
from summarization.abstractive.llama import LLamaSummarizer
from summarization.abstractive.mistral import MistralSummarizer
from summarization.abstractive.prophetnet import ProphetNetSummarizer
from summarization.abstractive.t5 import T5AbstractiveSummarizer
from summarization.extractive.bert import BertSummarizer
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.extractive.lexrank import LexRankSummarizer
from summarization.extractive.lsa import LSASummarizer
from summarization.extractive.mk_roberta import MkRobertaSummarizer
from summarization.extractive.sumarrnet import SummaRuNNerSummarizer
from summarization.extractive.textrank import TextRankerSummarizer
from summarization.hybrid.hybrid_summarizer import HybridSummarizer
from summarization.utils import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self, groq_client: Groq):
        self.extractive_models = {
            "textrank": TextRankerSummarizer(),
            "bert": BertSummarizer(),
            "lexrank": LexRankSummarizer(),
            "summarunner": SummaRuNNerSummarizer(),
            "lsa": LSASummarizer(),
            "mkroberta": MkRobertaSummarizer(),
        }
        self.abstractive_models = {
            "t5-small": T5AbstractiveSummarizer(
                extractive_summarizer=TextRankerSummarizer()
            ),
            "bart": BartSummarizer(),
            "distilbart": DistilBARTSummarizer(),
            "prophetnet": ProphetNetSummarizer(),
            "llama": LLamaSummarizer(groq_client=groq_client),
            "mistral": MistralSummarizer(groq_client=groq_client),
            "deepseek": DeepSeekSummarizer(groq_client=groq_client),
        }
        self.hybrid_models = {
            "hybrid": HybridSummarizer(text_ranker=self.extractive_models["textrank"], llama_summarizer=self.abstractive_models["llama"])
        }
        logger.info(f"Initialized summarizer")

    def generate_summary(
        self,
        text: str,
        summary_type: str = "abstractive",
        model_name: str = "t5-small",
        max_length: int = 150,
    ) -> str:
        """Generate summary from PDF file content based on specified summary type"""
        logger.info(f"Generating {summary_type} summary with {model_name} model")

        if summary_type.lower() == "extractive":
            extractive_summarizer: ExtractiveSummarizer = self.extractive_models.get(
                model_name
            )
            return extractive_summarizer.generate_extractive_summary(text)
        elif summary_type.lower() == "abstractive":
            abstractive_summarizer: AbstractiveSummarizer = self.abstractive_models.get(
                model_name
            )
            return abstractive_summarizer.generate_abstractive_summary(text=text)
        hybrid_summarizer: HybridSummarizer = self.hybrid_models.get(model_name)
        return hybrid_summarizer.generate_summary(text=text)

    def generate_pdf_summary(
        self,
        file_bytes: bytes,
        summary_type: str = "abstractive",
        model_name: str = "t5-small",
        max_length: int = 150,
    ) -> str:
        """Generate summary from PDF file content based on specified summary type"""
        logger.info(
            f"Generating {summary_type} summary for PDF with {model_name} model"
        )

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

    def get_summary_type_from_model(self, model_name: str) -> str:
        if model_name in self.abstractive_models.keys():
            return "abstractive"
        elif model_name in self.extractive_models.keys():
            return "extractive"
        elif model_name in self.hybrid_models.keys():
            return "hybrid"
        else:
            raise ValueError(f"Unknown model: {model_name}")
