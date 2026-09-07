from functools import partial
import os
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.utils import logging
from utils import compute_metrics, tokenize_and_align_labels

logging.set_verbosity_error()

# Setup and configuration
DATASET_PATH = "task1/data/processed_dataset"
MODEL_NAME = "dslim/bert-base-NER"
OUTPUT_DIR = "task1/model_weights"
REPO_ID = "TerentievaSof/bert-base-mountain-ner"

# Define label mappings
label2id = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}
id2label = {0: "O", 1:"B-MOUNTAIN", 2:"I-MOUNTAIN"}
labels = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]

def main():    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Run dataset_creation.ipynb first")

    # Load dataset and tokenizer
    dataset = Dataset.load_from_disk(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Train/val split
    dataset_split = dataset.train_test_split(test_size=0.2, seed=29)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    # Tokenize datasets
    tokenize_fn = partial(tokenize_and_align_labels, tokenizer=tokenizer)
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=val_dataset.column_names)
    
    # Load PyTorch Model
    model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics_fn = partial(compute_metrics, labels=labels)

    # Training Arguments & Trainer
    training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,  
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,  
    logging_steps=10,   
    lr_scheduler_type="reduce_lr_on_plateau",  
    warmup_ratio=0.1, 
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,    
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    # Saving locally
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Uploaded to Hugging Face Hub    
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    
    if hf_token:
        model.push_to_hub(REPO_ID, token=hf_token)
        tokenizer.push_to_hub(REPO_ID, token=hf_token)
    

# Run Training
if __name__ == "__main__":    
    main()

