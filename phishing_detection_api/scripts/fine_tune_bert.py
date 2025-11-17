import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import torch
import numpy as np
import optuna
import os

# --- Configuration ---
MODEL_NAME = 'distilbert-base-uncased'

def compute_metrics(p):
    """Computes accuracy and F1 score for evaluation."""
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted'),
    }

def model_init():
    """Initializes a new model for each trial."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def optuna_hp_space(trial: optuna.trial.Trial) -> dict:
    """Defines the search space for Optuna."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 2),  # <-- REDUCED EPOCHS
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8]),  # <-- REDUCED BATCH SIZE
        "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
    }

def fine_tune_bert_with_tuning():
    """
    Loads data, uses Optuna to find the best hyperparameters for DistilBERT,
    evaluates, and saves the final specialized model.
    """
    print("--- Starting BERT Hyperparameter Search & Fine-Tuning ---")
    set_seed(42)

    # --- Create robust, absolute paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    
    DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'phishing_text_dataset.csv')
    TUNER_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'ml_models', 'bert_tuner_checkpoints')
    FINAL_MODEL_PATH = os.path.join(PROJECT_ROOT, 'ml_models', 'my_finetuned_phishing_model')
    LOGGING_PATH = os.path.join(PROJECT_ROOT, 'logs')

    # --- 1. Load Your Custom Dataset ---
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: '{DATA_PATH}' not found. Please run 'create_datasets.py' first.")
        return

    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].astype(str)

    # --- 2. Prepare the Data for the Model ---
    train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
    train_df, eval_df = train_test_split(train_val_df, test_size=0.1, random_state=42, stratify=train_val_df['label'])

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    print("[*] Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # --- 3. Set Up Base Training Arguments ---
    base_training_args = TrainingArguments(
        output_dir=TUNER_CHECKPOINT_PATH,
        per_device_eval_batch_size=8,
        logging_dir=LOGGING_PATH,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,  # <-- ADDED FOR MEMORY
        report_to="none",               # <-- ADDED TO STOP WANDB
    )

    # --- 4. Initialize Trainer for Hyperparameter Search ---
    trainer = Trainer(
        model_init=model_init,
        args=base_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # --- 5. Run the Hyperparameter Search! ---
    print(f"[*] Starting hyperparameter search with {torch.cuda.device_count()} GPU(s)...")
    
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=2  # <-- REDUCED TRIALS
    )

    print("[*] Hyperparameter search complete.")
    print(f"--- Best Run Found ---")
    print(f"Score (eval_f1): {best_run.objective}")
    print("Best Hyperparameters:")
    print(best_run.hyperparameters)

    # --- 6. Train the Final Model with Best Parameters ---
    print("\n[*] Training final model with the best hyperparameters...")
    
    best_params = best_run.hyperparameters
    
    final_training_args = TrainingArguments(
        output_dir=FINAL_MODEL_PATH,
        num_train_epochs=best_params['num_train_epochs'],
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        per_device_eval_batch_size=base_training_args.per_device_eval_batch_size,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        gradient_accumulation_steps=2,  # <-- ADDED FOR MEMORY
        report_to="none",               # <-- ADDED TO STOP WANDB
    )

    final_trainer = Trainer(
        model=model_init(),
        args=final_training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    final_trainer.train()
    print("[*] Final model training complete.")

    # --- 7. Evaluate the Final Model on the Test Set ---
    print("\n--- Final Model Evaluation on Unseen Test Data ---")
    predictions = final_trainer.predict(tokenized_test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    
    print("Classification Report:")
    print(classification_report(test_df['label'], preds, target_names=['Safe (0)', 'Phishing (1)']))

    # --- 8. Save the Final, Best-Performing Model ---
    final_trainer.save_model(FINAL_MODEL_PATH)
    tokenizer.save_pretrained(FINAL_MODEL_PATH)
    print(f"✅ Best fine-tuned BERT model saved to '{FINAL_MODEL_PATH}'")

if __name__ == "__main__":
    fine_tune_bert_with_tuning()