import logging
import os
from functools import lru_cache

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import AutoTokenizer, pipeline

from summarization.utils import extract_text_from_pdf, split_text_into_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Summarizer:
    def __init__(self):
        self.extractive_summarizer = TextRankSummarizer()

    @staticmethod
    def get_latest_model(base_dir="./model_checkpoints"):
        """Find the most recently trained model"""
        if not os.path.exists(base_dir):
            logger.warning("No trained model found, using 't5-small'.")
            return "t5-small"

        model_dirs = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        return max(model_dirs, key=os.path.getmtime) if model_dirs else "t5-small"

    @lru_cache(maxsize=1)
    def load_abstractive_model(self):
        """Load and cache the abstractive summarization model"""
        model_name = self.get_latest_model()
        logger.info(f"Loading abstractive model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return pipeline("summarization", model=model_name, tokenizer=tokenizer)

    def generate_abstractive_summary_chunk(self, text: str, max_length=150):
        """Generate abstractive summary using the latest fine-tuned model"""
        try:
            summarizer = self.load_abstractive_model()
            if len(text.split()) < 10:
                logger.warning("Text too short for abstractive summarization")
                return text

            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min(50, max_length // 2),
                do_sample=False,
            )
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"Error in abstractive summarization: {e}")
            logger.info("Falling back to extractive summarization")
            return self.generate_extractive_summary(text)

    def generate_abstractive_summary(self, text: str) -> str:
        word_count = len(text.split())
        if word_count > 1000:
            logger.info(
                "Text is long; using hybrid summarization (extractive + abstractive)"
            )
            extractive = self.generate_extractive_summary(
                text, num_sentences_per_chunk=5
            )
            return self.generate_abstractive_summary_chunk(extractive)

        logger.info("Text is short; using 2-chunk abstractive summarization")
        chunk_summaries = []
        for chunk in split_text_into_chunks(text, 300):
            try:
                summary = self.generate_abstractive_summary_chunk(chunk)
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"Chunk summarization failed: {e}")

        combined = " ".join(chunk_summaries)
        if len(combined.split()) < 300:
            return combined
        return self.generate_abstractive_summary(combined)

    def generate_extractive_summary_chunk(self, text: str, num_sentences=3) -> str:
        """Generate extractive summary using TextRank"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary = self.extractive_summarizer(parser.document, num_sentences)

            if not summary:
                logger.warning("No sentences extracted, returning original text")
                return text[:500] + "..." if len(text) > 500 else text

            return " ".join(str(sentence) for sentence in summary)
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return text[:500] + "..." if len(text) > 500 else text

    def generate_extractive_summary(self, text: str, num_sentences_per_chunk=3):
        summaries = []
        for chunk in split_text_into_chunks(text, max_words=1000):
            summary = self.generate_extractive_summary_chunk(
                chunk, num_sentences_per_chunk
            )
            summaries.append(summary)

        final_summary = self.generate_extractive_summary_chunk(
            " ".join(summaries), num_sentences=5
        )

        return final_summary

    def generate_pdf_summary(
        self,
        file_bytes: bytes,
        summary_type: str = "abstractive",
        max_length: int = 150,
    ) -> str:
        """Generate summary from PDF file content based on specified summary type"""
        logger.info(f"Generating {summary_type} summary for PDF")

        text = extract_text_from_pdf(file_content=file_bytes)

        if not text or len(text.strip()) < 10:
            return "Insufficient text content found in the PDF for summarization."

        if summary_type.lower() == "extractive":
            num_sentences = max(3, len(text) // 500)
            summary = self.generate_extractive_summary(text, num_sentences)
        else:
            summary = self.generate_abstractive_summary(text=text)

        return summary
