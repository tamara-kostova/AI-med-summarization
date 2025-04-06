import logging

from rouge_score import rouge_scorer

from summarization.summarizer import Summarizer

logging.basicConfig(level=logging.INFO)


class Evaluator:
    def __init__(self, summarizer: Summarizer):
        self.summarizer = summarizer
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def evaluate_summary(self, reference: str, generated: str):
        """Evaluate a generated summary against a reference"""
        return self.scorer.score(reference, generated)

    def compare_summaries(self, text: str, reference_summary: str):
        """Compare extractive and abstractive summaries using ROUGE"""
        text = text[:2048]

        extractive_summary = self.summarizer.generate_extractive_summary(text)
        abstractive_summary = self.summarizer.generate_abstractive_summary(text)

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
            "recommended_method": recommended 
        }
