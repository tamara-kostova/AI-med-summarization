import logging 
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from summarization.extractive.extractive_summarizer import ExtractiveSummarizer

nltk.download("punkt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSASummarizer(ExtractiveSummarizer):
    def __init__(self, num_sentences: int = 5):
        self.num_sentences = num_sentences

    def generate_extractive_summary(self, text: str) -> str:
        sentences = sent_tokenize(text)
        if len(sentences) <= self.num_sentences:
            return text

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)

        svd = TruncatedSVD(
            n_components=min(self.num_sentences, tfidf_matrix.shape[1] - 1)
        )
        svd_matrix = svd.fit_transform(tfidf_matrix)

        sentence_scores = np.linalg.norm(svd_matrix, axis=1)

        ranked_sentence_indices = np.argsort(sentence_scores)[::-1][
            : self.num_sentences
        ]
        ranked_sentence_indices.sort()

        summary = " ".join([sentences[i] for i in ranked_sentence_indices])
        logger.info(f"Succesfully generated summary with model LSA")
        return summary
