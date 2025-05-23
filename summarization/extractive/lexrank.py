import logging
from typing import List

import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from summarization.extractive.extractive_summarizer import ExtractiveSummarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexRankSummarizer(ExtractiveSummarizer):
    def __init__(
        self,
        threshold: float = 0.1,
        damping: float = 0.85,
        epsilon: float = 1e-4,
        max_iter: int = 100,
    ):
        """
        Initialize the LexRank summarizer.
        """
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer data...")
            nltk.download("punkt")

        self.threshold = threshold
        self.damping = damping
        self.epsilon = epsilon
        self.max_iter = max_iter
        logger.info("LexRank summarizer initialized")

    def _create_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Create a similarity matrix for the input sentences using TF-IDF and cosine similarity.
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        similarity_matrix[similarity_matrix < self.threshold] = 0

        return similarity_matrix

    def _power_method(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the power iteration method to compute sentence scores.
        """
        n = similarity_matrix.shape[0]

        row_sums = similarity_matrix.sum(axis=1, keepdims=True)

        row_sums[row_sums == 0] = 1
        transition_matrix = similarity_matrix / row_sums

        scores = np.ones(n) / n

        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            scores = (1 - self.damping) / n + self.damping * (
                transition_matrix.T @ scores
            )

            if np.abs(scores - prev_scores).sum() < self.epsilon:
                break

        return scores

    def generate_extractive_summary(
        self,
        text: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 1500,
    ) -> str:
        """
        Generate an extractive summary for the given text.
        """
        logger.info("Generating LexRank extractive summary")

        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            logger.warning("Text too short for effective summarization")
            return text

        similarity_matrix = self._create_similarity_matrix(sentences)

        scores = self._power_method(similarity_matrix)

        num_sentences = max(1, min(int(len(sentences) * ratio), len(sentences)))

        ranked_indices = np.argsort(scores)[::-1][:num_sentences]
        selected_indices = sorted(ranked_indices)

        summary_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(summary_sentences)

        if len(summary) < min_length and len(sentences) > len(summary_sentences):
            remaining_indices = [
                i for i in range(len(sentences)) if i not in selected_indices
            ]
            remaining_indices_by_score = sorted(
                remaining_indices, key=lambda i: scores[i], reverse=True
            )

            for idx in remaining_indices_by_score:
                summary_sentences.append(sentences[idx])
                selected_indices.append(idx)
                selected_indices.sort()
                summary = " ".join([sentences[i] for i in selected_indices])

                if len(summary) >= min_length:
                    break

        if len(summary) > max_length:
            summary_sentences = []
            current_length = 0

            for i in selected_indices:
                sentence = sentences[i]
                if current_length + len(sentence) + 1 <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break

            summary = " ".join(summary_sentences)

        logger.info(f"Succesfully generated summary with model LexRank")
        return summary
