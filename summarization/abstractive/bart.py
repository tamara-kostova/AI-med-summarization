import logging
from functools import lru_cache

from transformers import AutoTokenizer, pipeline

from summarization.abstractive.abstractive_summarizer import AbstractiveSummarizer
from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.extractive.textrank import TextRankerSummarizer

logger = logging.getLogger(__name__)


class BartSummarizer(AbstractiveSummarizer):
    def __init__(
        self,
        model_name="facebook/bart-large-cnn",
        extractive_summarizer: ExtractiveSummarizer = TextRankerSummarizer(),
        max_input_length=1024  
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = pipeline(
            "summarization", model=model_name, tokenizer=self.tokenizer, device=-1
        )
        self.extractive_summarizer = extractive_summarizer

    @lru_cache(maxsize=1)
    def _load_model(self):
        return pipeline(
            "summarization",
            model=self.model_name,
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            device=-1
        )

    def generate_abstractive_summary(self, text: str, max_length: int = 150) -> str:
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

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_input_length)
            input_length = len(inputs["input_ids"][0])

            if input_length > self.max_input_length:
                logger.warning(f"Input text too long, truncating to {self.max_input_length} tokens")
                text = self.tokenizer.decode(inputs["input_ids"][0][:self.max_input_length], skip_special_tokens=True)

            summarizer = self._load_model()

            summary = summarizer(
                text,
                max_length=min(max_length, self.max_input_length),
                min_length=30,
                do_sample=False,
                truncation=True,
                no_repeat_ngram_size=3
            )
            if not summary or not isinstance(summary, list):
                logger.error("Invalid summary format from model")
                return "Summary generation failed"
            
            return summary[0]["summary_text"]
        except Exception as e:
            logger.error(f"BART summarization error: {e}")
            raise
