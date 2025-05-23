import os
import csv
from groq import Groq

from tqdm import tqdm

from summarization.summarizer import Summarizer
from summarization.evaluator import Evaluator
import pandas as pd
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "testing/sumpubmed_dataset"
TEXT_DIR = os.path.join(DATASET_ROOT, "text")
ABSTRACT_DIR = os.path.join(DATASET_ROOT, "abstract")
OUTPUT_CSV = "testing/sumpubmed_model_all_results.csv"
AVERAGES_OUTPUT_CSV = "testing/sumpubmed_model_average_all_scores.csv"
CHECKPOINT_FILE = "testing/evaluation_checkpoint.json"
BATCH_SIZE = 10


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


def save_checkpoint(results, processed_samples, checkpoint_file):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "processed_samples": processed_samples,
        "results": results,
    }
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)
    logger.info(f"Checkpoint saved with {len(results)} results")


def load_checkpoint(checkpoint_file):
    """Load previous progress from checkpoint file"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                f"Loaded checkpoint with {len(data['results'])} previous results"
            )
            return data["results"], set(data["processed_samples"])
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return [], set()
    return [], set()


def save_results_to_csv(results, output_file):
    """Save results to CSV file"""
    if not results:
        return

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "model",
                "type",
                "rouge1",
                "rouge2",
                "rougeL",
                "bleu1",
                "bleu2",
                "bleu4",
                "bertscore_precision",
                "bertscore_recall",
                "bertscore_f1",
                "paper_length",
                "summary_length",
                "reference_length",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def calculate_and_save_averages(results, averages_output_file):
    """Calculate and save average scores"""
    if not results:
        return

    results_df = pd.DataFrame(results)
    average_scores = (
        results_df.groupby("model")
        .agg(
            {
                "rouge1": "mean",
                "rouge2": "mean",
                "rougeL": "mean",
                "bleu1": "mean",
                "bleu2": "mean",
                "bleu4": "mean",
                "bertscore_precision": "mean",
                "bertscore_recall": "mean",
                "bertscore_f1": "mean",
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
            "bleu1": "avg_bleu1",
            "bleu2": "avg_bleu2",
            "bleu4": "avg_bleu4",
            "bertscore_precision": "avg_bertscore_precision",
            "bertscore_recall": "avg_bertscore_recall",  # Fixed typo
            "bertscore_f1": "avg_bertscore_f1",
            "paper_length": "avg_paper_length",
            "summary_length": "avg_summary_length",
            "reference_length": "avg_reference_length",
        },
        inplace=True,
    )

    average_scores.to_csv(averages_output_file, index=False)
    logger.info(f"Averages saved to {averages_output_file}")


def evaluate_single_sample(
    idx,
    text_dir,
    abstract_dir,
    summarizer,
    evaluator,
    abstractive_models,
    extractive_models,
):
    """Evaluate a single sample with all models"""

    def read_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    text_path = os.path.join(text_dir, f"text_{idx}.txt")
    abstract_path = os.path.join(abstract_dir, f"abst_{idx}.txt")

    text = read_file(text_path)
    reference = read_file(abstract_path)

    if len(text) < 50 or len(reference) < 10:
        logger.warning(f"Skipping sample {idx}: text or reference too short")
        return []

    sample_results = []

    for model in abstractive_models:
        try:
            summary = summarizer.generate_summary(
                text, summary_type="abstractive", model_name=model, max_length=150
            )
            scores = evaluator.evaluate_summary(reference, summary)
            logger.info(
                f"Sample {idx}, {model} (abstractive): ROUGE-1={scores['rouge1']:.3f}"
            )

            sample_results.append(
                {
                    "sample_id": idx,
                    "model": model,
                    "type": "abstractive",
                    "rouge1": scores["rouge1"],
                    "rouge2": scores["rouge2"],
                    "rougeL": scores["rougeL"],
                    "bleu1": scores["bleu1"],
                    "bleu2": scores["bleu2"],
                    "bleu4": scores["bleu4"],
                    "bertscore_precision": scores["bertscore_precision"],
                    "bertscore_recall": scores["bertscore_recall"],
                    "bertscore_f1": scores["bertscore_f1"],
                    "paper_length": len(text),
                    "summary_length": len(summary),
                    "reference_length": len(reference),
                }
            )
        except Exception as e:
            logger.error(f"Abstractive {model} on sample {idx}: {e}")

    for model in extractive_models:
        try:
            summary = summarizer.generate_summary(
                text, summary_type="extractive", model_name=model, max_length=150
            )
            scores = evaluator.evaluate_summary(reference, summary)
            logger.info(
                f"Sample {idx}, {model} (extractive): ROUGE-1={scores['rouge1']:.3f}"
            )

            sample_results.append(
                {
                    "sample_id": idx,
                    "model": model,
                    "type": "extractive",
                    "rouge1": scores["rouge1"],
                    "rouge2": scores["rouge2"],
                    "rougeL": scores["rougeL"],
                    "bleu1": scores["bleu1"],
                    "bleu2": scores["bleu2"],
                    "bleu4": scores["bleu4"],
                    "bertscore_precision": scores["bertscore_precision"],
                    "bertscore_recall": scores["bertscore_recall"],
                    "bertscore_f1": scores["bertscore_f1"],
                    "paper_length": len(text),
                    "summary_length": len(summary),
                    "reference_length": len(reference),
                }
            )
        except Exception as e:
            logger.error(f"Extractive {model} on sample {idx}: {e}")

    return sample_results


def evaluate_models(num_samples: int = 100, resume_from_checkpoint: bool = True):
    """Main evaluation function with checkpointing"""
    if resume_from_checkpoint:
        results, processed_samples = load_checkpoint(CHECKPOINT_FILE)
        logger.info(
            f"Resuming evaluation. Already processed {len(processed_samples)} samples."
        )
    else:
        results, processed_samples = [], set()
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            logger.info("Starting fresh evaluation (removed existing checkpoint)")

    sample_ids = get_sample_ids(TEXT_DIR, ABSTRACT_DIR, num_samples)
    remaining_samples = [sid for sid in sample_ids if sid not in processed_samples]

    logger.info(
        f"Total samples: {len(sample_ids)}, Remaining: {len(remaining_samples)}"
    )

    abstractive_models = [
        "t5-small",
        "bart",
        "distilbart",
        "prophetnet",
        "llama",
        "deepseek",
        "mistral",
    ]
    extractive_models = ["bert", "textrank", "lexrank", "summarunner", "lsa"]

    groq_client = Groq(
        api_key=""
    )
    summarizer = Summarizer(groq_client=groq_client)
    evaluator = Evaluator(summarizer=summarizer)

    batch_count = 0

    try:
        for i, idx in enumerate(tqdm(remaining_samples, desc="Evaluating samples")):
            try:
                sample_results = evaluate_single_sample(
                    idx,
                    TEXT_DIR,
                    ABSTRACT_DIR,
                    summarizer,
                    evaluator,
                    abstractive_models,
                    extractive_models,
                )

                results.extend(sample_results)
                processed_samples.add(idx)
                batch_count += 1

                if batch_count % BATCH_SIZE == 0:
                    save_checkpoint(results, list(processed_samples), CHECKPOINT_FILE)
                    save_results_to_csv(results, OUTPUT_CSV)
                    calculate_and_save_averages(results, AVERAGES_OUTPUT_CSV)
                    logger.info(f"Batch checkpoint: {batch_count} samples processed")

            except Exception as e:
                logger.error(f"Rrror processing sample {idx}: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
    finally:
        save_checkpoint(results, list(processed_samples), CHECKPOINT_FILE)
        save_results_to_csv(results, OUTPUT_CSV)
        calculate_and_save_averages(results, AVERAGES_OUTPUT_CSV)

        logger.info(f"Evaluation complete. Processed {len(processed_samples)} samples.")
        logger.info(f"Total results: {len(results)}")
        logger.info(f"Results saved to {OUTPUT_CSV}")
        logger.info(f"Averages saved to {AVERAGES_OUTPUT_CSV}")


if __name__ == "__main__":
    evaluate_models(num_samples=100, resume_from_checkpoint=True)
