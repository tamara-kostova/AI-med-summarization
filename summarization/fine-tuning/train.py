import os
from datetime import datetime

import aiohttp
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    BartForConditionalGeneration, 
    BartTokenizer
)


def train_t5_model(
    model_name="t5-small", output_dir="./model_checkpoints", num_epochs=3
):
    """
    Train a summarization t5 model

    Args:
        model_name (str): Pretrained model to fine-tune
        output_dir (str): Directory to save trained models
        num_epochs (int): Number of training epochs
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    dataset = load_dataset(
        "scientific_papers",
        "pubmed",
        split=["train[:5000]", "validation[:500]"],
        trust_remote_code=True,
        cache_dir="./data",
        download_mode="reuse_dataset_if_exists",
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    train_dataset, eval_dataset = dataset

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_data(examples):
        inputs = ["summarize: " + doc for doc in examples["article"]]
        model_inputs = tokenizer(
            inputs, max_length=512, truncation=True, padding="max_length"
        )
        labels = tokenizer(
            examples["abstract"], max_length=128, truncation=True, padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset_train = train_dataset.map(preprocess_data, batched=True)
    tokenized_dataset_validate = eval_dataset.map(preprocess_data, batched=True)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{model_output_dir}/logs",
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_validate,
    )

    trainer.train()

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f"Model trained and saved to: {model_output_dir}")
    return model_output_dir


def train_bart(model_name="facebook/bart-base", output_dir="./model_checkpoints_bart", num_epochs=3):
    """
    Train a BART model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"{model_name.split('/')[-1]}_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    dataset = load_dataset(
        "scientific_papers",
        "pubmed",
        split=["train[:5000]", "validation[:500]"],
        trust_remote_code=True,
        cache_dir="./data",
        download_mode="reuse_dataset_if_exists",
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    train_dataset, eval_dataset = dataset

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    def preprocess_data(examples):
        inputs = examples["article"]
        model_inputs = tokenizer(
            inputs, 
            max_length=512,
            truncation=True, 
            padding="max_length"
        )
        labels = tokenizer(
            examples["abstract"], 
            max_length=128,
            truncation=True, 
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset_train = train_dataset.map(preprocess_data, batched=True)
    tokenized_dataset_validate = eval_dataset.map(preprocess_data, batched=True)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{model_output_dir}/logs",
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adafactor",
        fp16=True,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        group_by_length=True,
        fp16_full_eval=True, 
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_validate,
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f"BART model trained and saved to: {model_output_dir}")
    return model_output_dir


if __name__ == "__main__":
    train_bart()
