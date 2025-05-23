import logging
from typing import Optional
from bert_score import score as bert_score
from rouge_score import rouge_scorer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from summarization.summarizer import Summarizer
from summarization.utils import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        logger.info(f"Initialized evaluator")

    def evaluate_summary(self, reference: str, generated: str):
        """Evaluate a generated summary against a reference"""
        rouge_scores = self.scorer.score(reference, generated)
        bleu_scores = self.compute_bleu(reference, generated)
        bertscore_scores = self.compute_bertscore(reference, generated)
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure,
            **bleu_scores,
            **bertscore_scores,
        }

    def compute_bleu(self, reference: str, candidate: str):
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        smoothie = SmoothingFunction().method4
        bleu1 = sentence_bleu(
            reference_tokens,
            candidate_tokens,
            weights=(1, 0, 0, 0),
            smoothing_function=smoothie,
        )
        bleu2 = sentence_bleu(
            reference_tokens,
            candidate_tokens,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothie,
        )
        bleu4 = sentence_bleu(
            reference_tokens,
            candidate_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        return {"bleu1": bleu1, "bleu2": bleu2, "bleu4": bleu4}

    def compute_bertscore(self, reference: str, candidate: str, lang: str = "en"):
        P, R, F1 = bert_score(
            [candidate], [reference], lang=lang, rescale_with_baseline=False
        )
        return {
            "bertscore_precision": P[0].item(),
            "bertscore_recall": R[0].item(),
            "bertscore_f1": F1[0].item(),
        }

    def generate_and_compare_summaries(
        self,
        text: str,
        reference_summary: str,
        extractive_model: str = "bert",
        abstractive_model: str = "t5-small",
    ):
        """Compare extractive and abstractive summaries using ROUGE"""
        # text = text[:2048]

        extractive_summary = self.summarizer.generate_summary(
            text, summary_type="extractive", model_name=extractive_model
        )
        abstractive_summary = self.summarizer.generate_summary(
            text, summary_type="abstractive", model_name=abstractive_model
        )

        extractive_scores = self.evaluate_summary(reference_summary, extractive_summary)
        abstractive_scores = self.evaluate_summary(
            reference_summary, abstractive_summary
        )
        extractive_rougeL = extractive_scores["rougeL"]
        abstractive_rougeL = abstractive_scores["rougeL"]

        if abstractive_rougeL > extractive_rougeL:
            recommended = "Abstractive"
        elif extractive_rougeL > abstractive_rougeL:
            recommended = "Extractive"
        else:
            recommended = "Either (Scores Equal)"

        return {
            "extractive_scores": extractive_scores,
            "abstractive_scores": abstractive_scores,
            "recommended_method": recommended,
        }

    def compare_summaries(self, summary_1: str, summary_2: str, reference_summary: str):

        scores1 = self.evaluate_summary(reference_summary, summary_1)
        scores2 = self.evaluate_summary(reference_summary, summary_2)
        rougeL_1 = scores1["rougeL"]
        rougeL_2 = scores2["rougeL"]

        if rougeL_1 > rougeL_2:
            recommended = "Model 1"
        elif rougeL_2 > rougeL_1:
            recommended = "Model 2"
        else:
            recommended = "Either (Scores Equal)"

        return {
            "model1_scores": scores1,
            "model2_scores": scores2,
            "recommended_method": recommended,
        }

    def pdf_summary(
        self,
        file_bytes,
        summary_type: str,
        model: str,
        compare_enabled,
        model2: Optional[str],
        reference_summary: Optional[str],
    ):
        summary1 = self.summarizer.generate_pdf_summary(file_bytes, summary_type, model)

        text = extract_text_from_pdf(file_bytes)

        result = {
            "summary": summary1,
            "summary_type": summary_type,
            "original_length": len(text),
            "summary_length": len(summary1),
        }
        if compare_enabled and model2:
            summary_type_model2 = self.summarizer.get_summary_type_from_model(model2)
            summary2 = self.summarizer.generate_pdf_summary(
                file_bytes, summary_type_model2, model2
            )

            evaluation = self.compare_summaries(
                reference_summary=reference_summary,
                summary_1=summary1,
                summary_2=summary2,
            )

            result.update(
                {
                    "comparison": {
                        "model1": model,
                        "model2": model2,
                        "summary2": summary2,
                        "scores": evaluation,
                    }
                }
            )
        return result
