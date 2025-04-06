from summarizer import Summarizer

from summarization.extractive.extractive_summarizer import ExtractiveSummarizer


class BertSummarizer(ExtractiveSummarizer):
    def __init__(self):
        self.model = Summarizer()

    def generate_extractive_summary(self, text):
        return self.model(text, min_length=60)
