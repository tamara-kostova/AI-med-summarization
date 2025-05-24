import logging
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from summarization.extractive.textrank import TextRankerSummarizer
from summarization.abstractive.llama import LLamaSummarizer
from summarization.utils import truncate_to_6000_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridSummarizer:
    def __init__(self, text_ranker: TextRankerSummarizer, llama_summarizer: LLamaSummarizer):
        self.text_ranker = text_ranker
        self.llama = llama_summarizer
        self.extractive_ratio = 0.3

    def extractive_stage(self, text, target_words=500):
        """
        Extract enough sentences using TextRank to reach at least target_words.
        """

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        sentences = list(parser.document.sentences)
        summary = []
        word_count = 0
        
        num_sentences = min(len(sentences), 20)
        while word_count < target_words and num_sentences <= len(sentences):
            summary = self.text_ranker.summarize(text, num_sentences=num_sentences)
            word_count = len(summary.split())
            num_sentences += 2
            if num_sentences > len(sentences):
                break
        return summary

    def prompt_llama_for_length(text, target_words=300):
        """
        Create a prompt for Llama to control the summary length.
        """
        return (
            f"You are an expert medical summarizer. "
            f"Summarize the following text in approximately {target_words} words, "
            f"preserving all key findings, methodology, and conclusions. "
            f"Make the summary clear and detailed, suitable for a medical professional.\n\n"
            f"Text:\n{text}"
        )

    def generate_summary(self, text: str) -> str:
        """Generate hybrid summary in two stages"""
        try:
            extractive = self.extractive_stage(text)

            prompt = self.prompt_llama_for_length(extractive)
            summary = self.llama.generate_abstractive_summary(prompt)
            summary = summary.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Hybrid summarization failed: {str(e)}")
            return "Summary generation error"
