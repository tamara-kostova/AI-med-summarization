import os
from datetime import datetime

import aiohttp
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
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


def train_bart(
    model_name="facebook/bart-base", output_dir="./model_checkpoints_bart", num_epochs=3
):
    """
    Train a BART model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(
        output_dir, f"{model_name.split('/')[-1]}_{timestamp}"
    )
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


def train_llama(
    model_name="openlm-research/open_llama_3b_v2",
    output_dir="./model_checkpoints_llama",
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"llama_summarizer_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "scientific_papers",
        "pubmed",
        split=[f"train[:{5000}]", f"validation[:{500}]"],
        trust_remote_code=True,
    )
    train_dataset, eval_dataset = dataset

    def format_for_llama(doc, summary):
        return f"""### Instruction: Summarize the following scientific article.

    ### Input:
    {doc}

    ### Response:
    {summary}"""

    def preprocess_data(examples):
        formatted_text = [
            format_for_llama(doc, summary)
            for doc, summary in zip(examples["article"], examples["abstract"])
        ]

        encodings = tokenizer(
            formatted_text,
            truncation=True,
            max_length=1024,
            padding="max_length",
            return_tensors="pt",
        )

        encodings["labels"] = encodings["input_ids"].clone()

        return encodings

    tokenized_train = train_dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_val = eval_dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    peft_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f"Llama model fine-tuned and saved to: {model_output_dir}")
    return model_output_dir


def train_distilbart(
    model_name="sshleifer/distilbart-cnn-6-6",
    output_dir="./model_checkpoints_bart",
    num_epochs=2,
):
    """
    Train a DistilBART model (306M parameters, distilled from 400M BART)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"distilbart_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    dataset = load_dataset(
        "scientific_papers",
        "pubmed",
        split=["train[:5000]", "validation[:500]"]
    )
    train_dataset, eval_dataset = dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{model_output_dir}/logs",
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adafactor",
        fp16=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        logging_steps=50,
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

    print(f"DistilBART model trained and saved to: {model_output_dir}")
    return model_output_dir


def train_prophetnet_small(
    model_name="microsoft/prophetnet-large-uncased",
    output_dir="./model_checkpoints_prophetnet",
    num_epochs=2,
):
    """
    Train a ProphetNet model for summarization
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = os.path.join(output_dir, f"prophetnet_small_{timestamp}")
    os.makedirs(model_output_dir, exist_ok=True)

    dataset = load_dataset(
        "scientific_papers",
        "pubmed",
        split=["train[:2000]", "validation[:200]"],
        trust_remote_code=True,
        cache_dir="./data",
        download_mode="reuse_dataset_if_exists",
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    train_dataset, eval_dataset = dataset

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

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

    tokenized_train = train_dataset.map(
        preprocess_data,
        batched=True,
    )

    tokenized_val = eval_dataset.map(
        preprocess_data,
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        num_train_epochs=num_epochs,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="adafactor",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()

    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print(f"ProphetNet small model fine-tuned and saved to: {model_output_dir}")
    return model_output_dir


if __name__ == "__main__":
    train_distilbart()
