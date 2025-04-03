import os
from datetime import datetime
from nlp import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)

def train_summarization_model(
    model_name="t5-small",
    output_dir="./model_checkpoints",
    num_epochs=3
):
    """
    Train a summarization model with comprehensive logging and saving
   
    Args:
        model_name (str): Pretrained model to fine-tune
        output_dir (str): Directory to save trained models
        num_epochs (int): Number of training epochs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        dataset = load_dataset(
            "scientific_papers", 
            "pubmed", 
            trust_remote_code=True,
            download_mode='reuse_dataset_if_exists'
        )
    except Exception as e:
        print(f"Initial dataset download failed: {e}")
        try:
            dataset = load_dataset(
                "scientific_papers", 
                "pubmed", 
                trust_remote_code=True,
                download_mode='force_redownload'
            )
        except Exception as download_error:
            print(f"Fallback download failed: {download_error}")
            print("Please check your internet connection and try again.")
            raise

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_data(examples):
        inputs = ["summarize: " + str(doc) for doc in examples["article"]]
        model_inputs = tokenizer(
            inputs,
            max_length=1024,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            [str(abstract) for abstract in examples["abstract"]],
            max_length=150,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Prepare the dataset
    tokenized_dataset = dataset.map(
        preprocess_data, 
        batched=True, 
        remove_columns=dataset["train"].column_names
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{model_output_dir}/logs",
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"]
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    print(f"Model trained and saved to: {model_output_dir}")
    return model_output_dir

if __name__ == "__main__":
    train_summarization_model()