import logging

from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from summarization.utils import split_text_into_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextRankerSummarizer:
    def __init__(self):
        self.summarizer = TextRankSummarizer()

    def summarize(self, text: str, num_sentences=3) -> str:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer(parser.document, num_sentences)
        return " ".join(str(sentence) for sentence in summary)

    def generate_extractive_summary_chunk(self, text: str, num_sentences=3) -> str:
        """Generate extractive summary using TextRank"""
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary = self.summarizer(parser.document, num_sentences)

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
