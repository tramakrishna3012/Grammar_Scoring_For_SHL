import pandas as pd
import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

def train_eval_model(train_csv_path, eval_csv_path):

    # 🔹 Load CSV with pandas, then convert to HuggingFace Dataset
    df = pd.read_csv(train_csv_path)
    train_dataset = Dataset.from_pandas(df)

    df = pd.read_csv(eval_csv_path)
    eval_dataset = Dataset.from_pandas(df)

    # 🔹 Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("vennify/t5-base-grammar-correction")

    # 🔹 Preprocessing function
    def preprocess(example):
        model_inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=64)
        labels = tokenizer(example["target"], padding="max_length", truncation=True, max_length=64)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 🔹 Tokenize entire dataset
    train_dataset = train_dataset.map(preprocess)
    val_dataset = eval_dataset.map(preprocess)

    # 🔹 Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained("vennify/t5-base-grammar-correction").to(device)

    # 🔹 Training arguments
    training_args = TrainingArguments(
        output_dir="./t5-finetuned-grammar",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    # 🔹 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # 🔹 Train the model
    trainer.train()

    # 🔹 Save the model
    model.save_pretrained("finetuned-t5-grammar")
    tokenizer.save_pretrained("finetuned-t5-grammar")

    return model, tokenizer