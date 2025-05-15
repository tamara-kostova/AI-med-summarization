import logging
import os
from functools import lru_cache

from transformers import AutoTokenizer, pipeline

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.extractive.textrank import TextRankerSummarizer

logger = logging.getLogger(__name__)


class DistilBARTSummarizer(AbstractiveSummarizer):
    """
    DistilBART CNN Summarizer with 306M parameters (smaller than BART)
    """

    def __init__(
        self,
        base_dir="model_checkpoints_bart",
        model_name="sshleifer/distilbart-cnn-6-6",
        extractive_summarizer: ExtractiveSummarizer = TextRankerSummarizer(),
        max_input_length=1024,
    ):
        self.base_dir = base_dir
        self.model_name = model_name
        self.extractive_summarizer = extractive_summarizer
        self._model = None
        self._tokenizer = None
        self.max_input_length = max_input_length

    @staticmethod
    def get_latest_model(base_dir="model_checkpoints_bart"):
        """Find the most recently trained DistilBART model"""
        if not os.path.exists(base_dir):
            logger.warning(
                "No trained model found, using 'sshleifer/distilbart-cnn-6-6'."
            )
            return "sshleifer/distilbart-cnn-6-6"

        model_dirs = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and "distilbart" in d
        ]
        return (
            max(model_dirs, key=os.path.getmtime)
            if model_dirs
            else "sshleifer/distilbart-cnn-6-6"
        )

    @lru_cache(maxsize=1)
    def _load_model(self):
        model_path = self.get_latest_model(self.base_dir)
        logger.info(f"Loading DistilBART model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return pipeline(
            "summarization", model=model_path, tokenizer=tokenizer, device=-1
        )

    def generate_abstractive_summary(self, text: str, max_length: int = 150) -> str:
        """Generate an abstractive summary using the DistilBART model"""
        try:
            logger.info(f"Input text length: {len(text.split())} words")
            if not text.strip():
                logger.warning("Empty text received for summarization")
                return ""
            if len(text.split()) < 10:
                logger.warning("Text too short for abstractive summarization")
                return text

            if self.extractive_summarizer and len(text.split()) > 1000:
                extractive = self.extractive_summarizer.generate_extractive_summary(
                    text
                )
                return self.generate_abstractive_summary(extractive)

            tokenizer = AutoTokenizer.from_pretrained(
                self.get_latest_model(self.base_dir)
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_input_length,
            )
            input_length = len(inputs["input_ids"][0])

            if input_length > self.max_input_length:
                logger.warning(
                    f"Input text too long, truncating to {self.max_input_length} tokens"
                )
                text = tokenizer.decode(
                    inputs["input_ids"][0][: self.max_input_length],
                    skip_special_tokens=True,
                )

            summarizer = self._load_model()

            summary = summarizer(
                text,
                max_length=min(max_length, self.max_input_length),
                min_length=30,
                do_sample=False,
                truncation=True,
                no_repeat_ngram_size=3,
            )
            if not summary or not isinstance(summary, list):
                logger.error("Invalid summary format from model")
                return "Summary generation failed"

            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"DistilBART summarization error: {e}")
            raise

    def _chunked_summarization(self, words):
        """Handle long texts by chunking them"""
        chunk_size = 600
        chunks = [
            " ".join(words[i : i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            try:
                summary = self._model(
                    chunk,
                    max_length=100,
                    min_length=20,
                    do_sample=False,
                    clean_up_tokenization_spaces=True,
                )
                chunk_summaries.append(summary[0]["summary_text"])
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                if self.extractive_summarizer:
                    extract = self.extractive_summarizer.generate_extractive_summary(
                        chunk
                    )
                    chunk_summaries.append(extract)

        combined = " ".join(chunk_summaries)

        if len(combined.split()) > 600:
            logger.info("Re-summarizing the combined chunk summaries")
            try:
                final_summary = self._model(
                    combined,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    clean_up_tokenization_spaces=True,
                )
                return final_summary[0]["summary_text"]
            except Exception as e:
                logger.error(f"Error re-summarizing combined chunks: {e}")
                if self.extractive_summarizer:
                    return self.extractive_summarizer.generate_extractive_summary(
                        combined
                    )
                return combined[:500] + "..."

        return combined
