import logging
from functools import lru_cache
import os

from transformers import AutoTokenizer, pipeline
from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer


logger = logging.getLogger(__name__)

class DistilBARTSummarizer(AbstractiveSummarizer):
    """
    DistilBART CNN Summarizer with 306M parameters (smaller than BART)
    """
    def __init__(
        self,
        model_name="sshleifer/distilbart-cnn-6-6",
        extractive_summarizer=None
    ):
        self.model_name = model_name
        self.extractive_summarizer = extractive_summarizer
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load the model and tokenizer only when needed"""
        if self._model is None:
            logger.info(f"Loading DistilBART model: {self.model_name}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = pipeline(
                    "summarization",
                    model=self.model_name,
                    tokenizer=self._tokenizer,
                    device=-1  # Use CPU
                )
                logger.info("DistilBART model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading DistilBART model: {e}")
                raise

    def generate_abstractive_summary(self, text: str) -> str:
        """Generate an abstractive summary using the DistilBART model"""
        if len(text.strip()) < 30:
            logger.warning("Text too short for summarization")
            return text

        self._load_model()
        
        # Process in chunks for long text
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        if tokens.input_ids.shape[1] >= 1000:
            logger.info("Text is very long; using chunked summarization")
            words = text.split()
            return self._chunked_summarization(words)
        
        logger.info("Processing text in single chunk")
        try:
            summary = self._model(
                text,
                max_length=150,
                min_length=30,
                do_sample=False,
                clean_up_tokenization_spaces=True
            )
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"Error in DistilBART summarization: {e}")
            if self.extractive_summarizer:
                logger.info("Falling back to extractive summarization")
                return self.extractive_summarizer.generate_extractive_summary(text)
            return text[:200] + "..."
            
    def _chunked_summarization(self, words):
        """Handle long texts by chunking them"""
        chunk_size = 600  # DistilBART can handle larger chunks
        chunks = [
            " ".join(words[i:i + chunk_size])
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
                    clean_up_tokenization_spaces=True
                )
                chunk_summaries.append(summary[0]["summary_text"])
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                if self.extractive_summarizer:
                    extract = self.extractive_summarizer.generate_extractive_summary(chunk)
                    chunk_summaries.append(extract)
        
        combined = " ".join(chunk_summaries)
        
        # If the combined summaries are still long, re-summarize
        if len(combined.split()) > 600:
            logger.info("Re-summarizing the combined chunk summaries")
            try:
                final_summary = self._model(
                    combined,
                    max_length=150,
                    min_length=50,
                    do_sample=False,
                    clean_up_tokenization_spaces=True
                )
                return final_summary[0]["summary_text"]
            except Exception as e:
                logger.error(f"Error re-summarizing combined chunks: {e}")
                if self.extractive_summarizer:
                    return self.extractive_summarizer.generate_extractive_summary(combined)
                return combined[:500] + "..."
        
        return combined