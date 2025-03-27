from rouge_score import rouge_scorer
from summarization.abstractive import generate_abstractive_summary
from summarization.extractive import generate_extractive_summary

def evaluate_summary(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)

def compare_summaries(text, reference_summary):
    text = text[:2048]
    extractive = generate_extractive_summary(text)
    abstractive = generate_abstractive_summary(text)

    extractive_scores = evaluate_summary(reference_summary, extractive)
    abstractive_scores = evaluate_summary(reference_summary, abstractive)

    return {
        "extractive_rouge": {
            "rouge1": extractive_scores['rouge1'].fmeasure,
            "rouge2": extractive_scores['rouge2'].fmeasure,
            "rougeL": extractive_scores['rougeL'].fmeasure
        },
        "abstractive_rouge": {
            "rouge1": abstractive_scores['rouge1'].fmeasure,
            "rouge2": abstractive_scores['rouge2'].fmeasure,
            "rougeL": abstractive_scores['rougeL'].fmeasure
        }
    }
