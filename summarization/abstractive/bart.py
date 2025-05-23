import logging
import os
from functools import lru_cache

from transformers import AutoTokenizer, pipeline

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.extractive.textrank import TextRankerSummarizer

logger = logging.getLogger(__name__)


class BartSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        base_dir="./model_checkpoints_bart",
        default_model="facebook/bart-large-cnn",
        extractive_summarizer: ExtractiveSummarizer = TextRankerSummarizer(),
        max_input_length=1024,
    ):
        self.base_dir = base_dir
        self.default_model = default_model
        self.max_input_length = max_input_length
        self.extractive_summarizer = extractive_summarizer

    @staticmethod
    def get_latest_model(base_dir="./model_checkpoints_bart"):
        """Find the most recently trained BART model"""
        if not os.path.exists(base_dir):
            logger.warning("No trained model found, using 'facebook/bart-large-cnn'.")
            return "facebook/bart-large-cnn"

        model_dirs = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        return (
            max(model_dirs, key=os.path.getmtime)
            if model_dirs
            else "facebook/bart-large-cnn"
        )

    @lru_cache(maxsize=1)
    def _load_model(self):
        model_path = self.get_latest_model(self.base_dir)
        logger.info(f"Loading BART model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return pipeline(
            "summarization", model=model_path, tokenizer=tokenizer, device=-1
        )

    def generate_abstractive_summary(self, text: str, max_length: int = 300) -> str:
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
            logger.info(f"Successfully generated summary with model Bart.")
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"BART summarization error: {e}")
            raise
