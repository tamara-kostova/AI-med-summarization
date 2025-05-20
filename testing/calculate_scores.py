import os
import csv
from tqdm import tqdm

from summarization.summarizer import Summarizer
from summarization.evaluator import Evaluator
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "testing/sumpubmed_dataset"
TEXT_DIR = os.path.join(DATASET_ROOT, "text")
ABSTRACT_DIR = os.path.join(DATASET_ROOT, "abstract")
OUTPUT_CSV = "testing/sumpubmed_model_rouge_results.csv"
AVERAGES_OUTPUT_CSV = "/sumpubmed_model_average_rouge_scores.csv"


def get_sample_ids(text_dir, abstract_dir, num_samples: int = 10):
    text_files = set(
        f
        for f in os.listdir(text_dir)[:num_samples]
        if f.startswith("text_") and f.endswith(".txt")
    )
    abstract_files = set(
        f
        for f in os.listdir(abstract_dir)[:num_samples]
        if f.startswith("abst_") and f.endswith(".txt")
    )
    ids = []
    for tf in text_files:
        idx = tf.replace("text_", "").replace(".txt", "")
        if f"abst_{idx}.txt" in abstract_files:
            ids.append(idx)
    return sorted(ids)


def evaluate_models(num_samples: int = 10):
    sample_ids = get_sample_ids(TEXT_DIR, ABSTRACT_DIR, num_samples)

    abstractive_models = [
        "t5-small",
        "bart",
        "distilbart",
        "prophetnet",
        # , "llama", "deepseek", "mistral"
    ]
    extractive_models = ["bert", "textrank", "lexrank", "summarunner", "lsa"]

    summarizer = Summarizer()
    evaluator = Evaluator(summarizer=summarizer)

    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    results = []

    for idx in tqdm(sample_ids, desc="Evaluating samples"):
        text_path = os.path.join(TEXT_DIR, f"text_{idx}.txt")
        abstract_path = os.path.join(ABSTRACT_DIR, f"abst_{idx}.txt")

        text = read_file(text_path)
        reference = read_file(abstract_path)

        if len(text) < 50 or len(reference) < 10:
            continue

        for model in abstractive_models:
            try:
                summary = summarizer.generate_summary(
                    text, summary_type="abstractive", model_name=model, max_length=150
                )
                scores = evaluator.evaluate_summary(reference, summary)
                results.append(
                    {
                        "sample_id": idx,
                        "model": model,
                        "type": "abstractive",
                        "rouge1": scores["rouge1"].fmeasure,
                        "rouge2": scores["rouge2"].fmeasure,
                        "rougeL": scores["rougeL"].fmeasure,
                        "paper_length": len(text),
                        "summary_length": len(summary),
                        "reference_length": len(reference),
                    }
                )
            except Exception as e:
                logger.error(f"Abstractive {model} on {idx}: {e}")

        for model in extractive_models:
            try:
                summary = summarizer.generate_summary(
                    text, summary_type="extractive", model_name=model, max_length=150
                )
                scores = evaluator.evaluate_summary(reference, summary)
                results.append(
                    {
                        "sample_id": idx,
                        "model": model,
                        "type": "extractive",
                        "rouge1": scores["rouge1"].fmeasure,
                        "rouge2": scores["rouge2"].fmeasure,
                        "rougeL": scores["rougeL"].fmeasure,
                        "paper_length": len(text),
                        "summary_length": len(summary),
                        "reference_length": len(reference),
                    }
                )
            except Exception as e:
                logger.error(f"Extractive {model} on {idx}: {e}")

    with open(
        OUTPUT_CSV, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "model",
                "type",
                "rouge1",
                "rouge2",
                "rougeL",
                "paper_length",
                "summary_length",
                "reference_length",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    logger.info(
        f"Evaluation complete. Results saved to {OUTPUT_CSV}"
    )

    results_df = pd.DataFrame(results)
    average_scores = (
        results_df.groupby("model")
        .agg(
            {
                "rouge1": "mean",
                "rouge2": "mean",
                "rougeL": "mean",
                "paper_length": "mean",
                "summary_length": "mean",
                "reference_length": "mean",
            }
        )
        .reset_index()
    )

    average_scores.rename(
        columns={
            "rouge1": "avg_rouge1",
            "rouge2": "avg_rouge2",
            "rougeL": "avg_rougeL",
            "paper_length": "avg_paper_length",
            "summary_length": "avg_summary_length",
            "reference_length": "avg_reference_length",
        },
        inplace=True,
    )

    average_scores.to_csv(AVERAGES_OUTPUT_CSV, index=False)

    logger.info(f"Averages saved to {AVERAGES_OUTPUT_CSV}")
    logger.info(average_scores)


if __name__ == "__main__":
    evaluate_models(num_samples=1)
