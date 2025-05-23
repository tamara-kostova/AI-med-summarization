import logging
from typing import List

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
from transformers import AutoModel, AutoTokenizer

from summarization.extractive.extractive_summarizer import ExtractiveSummarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaRuNNer(nn.Module):
    """
    Neural network model for extractive summarization.
    """

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1
    ):
        super(SummaRuNNer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.doc_rep = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

        self.content = nn.Linear(2 * hidden_dim, 1)

        self.salience = nn.Bilinear(2 * hidden_dim, 2 * hidden_dim, 1)

        self.novelty = nn.Bilinear(2 * hidden_dim, 2 * hidden_dim, 1)

        self.pos_emb = nn.Embedding(500, 1)

        self.classifier = nn.Linear(1, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sent_embeddings, sentence_lengths, doc_length):
        """
        Forward pass for SummaRuNNer model
        """
        batch_size, max_doc_len, _ = sent_embeddings.size()

        packed_sents = nn.utils.rnn.pack_padded_sequence(
            sent_embeddings, sentence_lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_sents)
        sent_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )

        doc_mask = (
            torch.arange(max_doc_len)
            .expand(batch_size, max_doc_len)
            .to(sent_output.device)
        )
        doc_mask = doc_mask < doc_length.unsqueeze(1)
        masked_output = sent_output * doc_mask.unsqueeze(2).float()

        doc_embedding = self.doc_rep(
            torch.sum(masked_output, dim=1) / doc_length.unsqueeze(1).float()
        )

        content_scores = self.content(sent_output)

        salience_scores = self.salience(
            sent_output, doc_embedding.unsqueeze(1).expand(-1, max_doc_len, -1)
        )

        position_idx = (
            torch.arange(max_doc_len)
            .expand(batch_size, max_doc_len)
            .to(sent_output.device)
        )
        position_scores = self.pos_emb(position_idx)

        novelty_scores = torch.zeros_like(content_scores)
        sum_embeddings = torch.zeros_like(doc_embedding)

        for i in range(max_doc_len):
            if i > 0:
                sum_embeddings = sum_embeddings + sent_output[:, i - 1, :] * probs[
                    :, i - 1
                ].unsqueeze(1)
                novelty_scores[:, i, :] = self.novelty(
                    sent_output[:, i, :].unsqueeze(1), sum_embeddings.unsqueeze(1)
                ).squeeze(1)

            scores = (
                content_scores[:, i, :]
                + salience_scores[:, i, :]
                + novelty_scores[:, i, :]
                + position_scores[:, i, :]
            )

            p = torch.sigmoid(self.classifier(scores))

            if i == 0:
                probs = p
            else:
                probs = torch.cat([probs, p], dim=1)

        return probs


class SummaRuNNerSummarizer(ExtractiveSummarizer):
    """
    Implementation of SummaRuNNer extractive summarization using pretrained sentence embeddings.
    """

    def __init__(self, model_name="bert-base-uncased", device=None, threshold=0.5):
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("corpora/stopwords")
        except LookupError:
            logger.info("Downloading NLTK resources...")
            nltk.download("punkt")
            nltk.download("stopwords")

        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = threshold

        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.content_weight = 1.0
        self.salience_weight = 1.0
        self.novelty_weight = 0.5
        self.position_weight = 0.1

        logger.info(f"SummaRuNNer summarizer initialized on {self.device}")

    def _get_sentence_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """
        Get sentence embeddings using pretrained language model.
        """
        embeddings = []

        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                outputs = self.model(**inputs)

                sentence_embedding = (
                    outputs.last_hidden_state.mean(dim=1).squeeze().cpu()
                )
                embeddings.append(sentence_embedding)

        return torch.stack(embeddings)

    def generate_extractive_summary(
        self,
        text: str,
        ratio: float = 0.2,
        min_length: int = 40,
        max_length: int = 1500,
    ) -> str:
        """
        Generate an extractive summary for the given text using SummaRuNNer principles.
        """
        logger.info("Generating SummaRuNNer extractive summary")

        sentences = sent_tokenize(text)
        if len(sentences) < 3:
            logger.warning("Text too short for effective summarization")
            return text
        sentence_embeddings = self._get_sentence_embeddings(sentences)

        doc_embedding = sentence_embeddings.mean(dim=0)

        scores = []
        summary_embedding = torch.zeros_like(doc_embedding)

        for i, sent_embedding in enumerate(sentence_embeddings):
            content_score = torch.norm(sent_embedding)

            salience_score = F.cosine_similarity(
                sent_embedding.unsqueeze(0), doc_embedding.unsqueeze(0)
            ).item()

            position_score = np.exp(-i / len(sentences))

            if i > 0 and torch.norm(summary_embedding) > 0:
                novelty_score = (
                    1
                    - F.cosine_similarity(
                        sent_embedding.unsqueeze(0), summary_embedding.unsqueeze(0)
                    ).item()
                )
            else:
                novelty_score = 1.0

            score = (
                self.content_weight * content_score
                + self.salience_weight * salience_score
                + self.novelty_weight * novelty_score
                + self.position_weight * position_score
            )

            scores.append(score)

            if score > self.threshold:
                summary_embedding = summary_embedding + sent_embedding

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

        logger.info(f"Succesfully generated summary with model SummaRuNNer")
        return summary
