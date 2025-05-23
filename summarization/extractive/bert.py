import logging

from summarizer import Summarizer

from summarization.extractive.extractive_summarizer import ExtractiveSummarizer
from summarization.utils import split_text_into_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BertSummarizer(ExtractiveSummarizer):
    def __init__(self, max_sentences=10, min_sentences=2, ratio=0.075):
        self.model = Summarizer()
        self.max_sentences = max_sentences
        self.min_sentences = min_sentences
        self.ratio = ratio

    def generate_extractive_summary(self, text):
        return self.model(
            text, min_length=60, num_sentences=self.max_sentences, ratio=self.ratio
        )

    def generate_extractive_summary_chunk(self, text: str) -> str:
        """Generate extractive summary using TextRank"""
        try:
            summary = self.model(text, min_length=60)

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
            summary = self.generate_extractive_summary_chunk(chunk)
            summaries.append(summary)

        final_summary = self.generate_extractive_summary_chunk(" ".join(summaries))

        logger.info(f"Successfully generated summary with model Bert.")
        return final_summary
