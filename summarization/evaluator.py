import logging
from typing import Optional

from rouge_score import rouge_scorer

from summarization.summarizer import Summarizer
from summarization.utils import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)


class Evaluator:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        print(f"Initialized evaluator")

    def evaluate_summary(self, reference: str, generated: str):
        """Evaluate a generated summary against a reference"""
        return self.scorer.score(reference, generated)

    def generate_and_compare_summaries(
        self,
        text: str,
        reference_summary: str,
        extractive_model: str = "bert",
        abstractive_model: str = "t5-small",
    ):
        """Compare extractive and abstractive summaries using ROUGE"""
        text = text[:2048]

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
        extractive_rougeL = extractive_scores["rougeL"].fmeasure
        abstractive_rougeL = abstractive_scores["rougeL"].fmeasure

        if abstractive_rougeL > extractive_rougeL:
            recommended = "Abstractive"
        elif extractive_rougeL > abstractive_rougeL:
            recommended = "Extractive"
        else:
            recommended = "Either (Scores Equal)"

        return {
            "extractive_rouge": {
                "rouge1": extractive_scores["rouge1"].fmeasure,
                "rouge2": extractive_scores["rouge2"].fmeasure,
                "rougeL": extractive_scores["rougeL"].fmeasure,
            },
            "abstractive_rouge": {
                "rouge1": abstractive_scores["rouge1"].fmeasure,
                "rouge2": abstractive_scores["rouge2"].fmeasure,
                "rougeL": abstractive_scores["rougeL"].fmeasure,
            },
            "recommended_method": recommended,
        }

    def compare_summaries(self, summary_1: str, summary_2: str, reference_summary: str):

        scores1 = self.evaluate_summary(reference_summary, summary_1)
        scores2 = self.evaluate_summary(reference_summary, summary_2)
        rougeL_1 = scores1["rougeL"].fmeasure
        rougeL_2 = scores2["rougeL"].fmeasure

        if rougeL_1 > rougeL_2:
            recommended = "Model 1"
        elif rougeL_2 > rougeL_1:
            recommended = "Model 2"
        else:
            recommended = "Either (Scores Equal)"

        return {
            "model1_rouge": {
                "rouge1": scores1["rouge1"].fmeasure,
                "rouge2": scores1["rouge2"].fmeasure,
                "rougeL": scores1["rougeL"].fmeasure,
            },
            "model2_rouge": {
                "rouge1": scores2["rouge1"].fmeasure,
                "rouge2": scores2["rouge2"].fmeasure,
                "rougeL": scores2["rougeL"].fmeasure,
            },
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
            summary2 = self.summarizer.generate_pdf_summary(
                file_bytes, summary_type, model2
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
                        "rouge_scores": evaluation,
                    }
                }
            )

        return result
