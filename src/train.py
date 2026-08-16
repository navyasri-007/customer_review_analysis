import os
import sys
import json
import argparse
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# ============================================================
# PROJECT & MODULE PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.data_preprocessing import prepare_dataset
except ImportError:
    from data_preprocessing import prepare_dataset

MODEL_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
    "deberta",
    "final"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


# ============================================================
# PYTORCH DATASET WRAPPER
# ============================================================

class ReviewDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train(
    model_name="microsoft/deberta-v3-base",
    epochs=2,
    batch_size=8,
    eval_batch_size=8,
    learning_rate=2e-5,
    sample_size_per_class=10000,
    max_length=128,
    output_dir=MODEL_OUTPUT_DIR
):
    print("=" * 60)
    print("CUSTOMER REVIEW SENTIMENT - DeBERTa-v3 TRAINING")
    print("=" * 60)

    # 1. Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Name      : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  Running on CPU. Consider using GPU for faster transformer training.")

    # 2. Load Dataset
    print(f"\n1. Preparing balanced dataset ({sample_size_per_class:,} per class)...")
    train_df, val_df, test_df = prepare_dataset(max_per_class=sample_size_per_class)

    # 3. Load Tokenizer
    print(f"\n2. Loading tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 4. Tokenization
    print("3. Tokenizing datasets...")

    def tokenize_data(df):
        return tokenizer(
            df["review"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length
        )

    train_encodings = tokenize_data(train_df)
    val_encodings = tokenize_data(val_df)
    test_encodings = tokenize_data(test_df)

    train_dataset = ReviewDataset(train_encodings, train_df["label"].tolist())
    val_dataset = ReviewDataset(val_encodings, val_df["label"].tolist())
    test_dataset = ReviewDataset(test_encodings, test_df["label"].tolist())

    print(f"  -> Train examples     : {len(train_dataset):,}")
    print(f"  -> Validation examples: {len(val_dataset):,}")
    print(f"  -> Test examples       : {len(test_dataset):,}")

    # 5. Load Model Architecture
    print(f"\n4. Loading model architecture: '{model_name}' (num_labels=2)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 6. Training Configuration
    # Handle both modern 'eval_strategy' and legacy 'evaluation_strategy'
    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "fp16": torch.cuda.is_available(),
        "logging_steps": 50,
        "report_to": "none"
    }

    try:
        training_args = TrainingArguments(eval_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)

    # 7. Trainer Setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 8. Train Model
    print("\n5. Starting DeBERTa fine-tuning...")
    train_result = trainer.train()
    print("\nTraining completed successfully!")

    # 9. Save Best Model & Tokenizer
    print(f"\n6. Saving final model and tokenizer to:\n  {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer persisted.")

    # 10. Evaluate on Test Dataset
    print("\n7. Evaluating on holdout test dataset...")
    test_results = trainer.evaluate(test_dataset)

    print("\n" + "=" * 60)
    print("FINAL TEST METRICS")
    print("=" * 60)
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"  {key:<25}: {value:.4f}")
        else:
            print(f"  {key:<25}: {value}")

    # 11. Save Summary Metrics to File
    summary_path = os.path.join(RESULTS_DIR, "training_metrics.json")
    training_summary = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "sample_size_per_class": sample_size_per_class,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_loss": float(train_result.training_loss) if hasattr(train_result, "training_loss") else None,
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_results.items()}
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4)
    print(f"\nSaved training summary to:\n  {summary_path}")

    return training_summary


# ============================================================
# CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeBERTa-v3 Sentiment Model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device train batch size (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Per device eval batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--sample_size", type=int, default=10000, help="Reviews per class (default: 10000)")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Base model checkpoint")
    parser.add_argument("--quick_test", action="store_true", help="Run quick 1-epoch test on 200 samples")

    args = parser.parse_args()

    if args.quick_test:
        print("[Quick Test Mode Enabled]")
        train(
            model_name=args.model_name,
            epochs=1,
            batch_size=4,
            eval_batch_size=4,
            learning_rate=args.learning_rate,
            sample_size_per_class=100
        )
    else:
        train(
            model_name=args.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            sample_size_per_class=args.sample_size,
            model_name=args.model_name
        )