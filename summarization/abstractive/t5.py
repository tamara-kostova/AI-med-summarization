import logging
import os
from functools import lru_cache

from transformers import AutoTokenizer, pipeline

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.extractive.textrank import TextRankSummarizer
from summarization.utils import split_text_into_chunks

logger = logging.getLogger(__name__)


class T5AbstractiveSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        base_dir="./model_checkpoints",
        default_model="t5-small",
        extractive_summarizer: ExtractiveSummarizer = TextRankSummarizer(),
    ):
        self.base_dir = base_dir
        self.default_model = default_model
        self.extractive_summarizer = extractive_summarizer

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

    def generate_abstractive_summary_chunk(self, text: str, max_length=300):
        """Generate abstractive summary using the latest fine-tuned model"""
        try:
            summarizer = self.load_abstractive_model()
            tokenizer = summarizer.tokenizer
            inputs = tokenizer(
                text, return_tensors="pt", max_length=512, truncation=True
            )
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] > 512:
                logger.warning("Text too long even after truncation")

            if len(text.split()) < 10:
                logger.warning("Text too short for abstractive summarization")
                return text
            tokenizer = summarizer.tokenizer
            tokens = tokenizer.encode(text)
            if len(tokens) > 512:
                logger.info(
                    f"Text exceeds model's token limit ({len(tokens)} > 512). Truncating..."
                )
                text = tokenizer.decode(tokens[:512], skip_special_tokens=True)
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
            return self.extractive_summarizer.generate_extractive_summary(text)

    def generate_abstractive_summary(self, text: str) -> str:
        word_count = len(text.split())
        if word_count > 1000:
            logger.info(
                "Text is long; using hybrid summarization (extractive + abstractive)"
            )
            extractive = self.extractive_summarizer.generate_extractive_summary(
                text, num_sentences_per_chunk=5
            )
            if len(extractive.split()) > 300:
                chunks = split_text_into_chunks(extractive, 250)
                chunk_summaries = []
                for chunk in chunks:
                    try:
                        summary = self.generate_abstractive_summary_chunk(chunk)
                        chunk_summaries.append(summary)
                    except Exception as e:
                        logger.error(f"Chunk summarization failed: {e}")
                combined = " ".join(chunk_summaries)
                if len(combined.split()) > 300:
                    return self.generate_abstractive_summary_chunk(combined)
                logger.info(f"Successfully generated summary with model T5.")
                return combined
            else:
                logger.info(f"Successfully generated summary with model T5.")
                return self.generate_abstractive_summary_chunk(extractive)

        logger.info("Text is short; using 2-chunk abstractive summarization")
        chunk_summaries = []
        for chunk in split_text_into_chunks(text, 300):
            try:
                summary = self.generate_abstractive_summary_chunk(chunk)
                chunk_summaries.append(summary)
            except Exception as e:
                logger.error(f"Chunk summarization failed: {e}")

        combined = " ".join(chunk_summaries)
        if len(combined.split()) < 300:
            logger.info(f"Successfully generated summary with model T5.")
            return combined
        logger.info(f"Successfully generated summary with model T5.")
        return self.generate_abstractive_summary(combined)
