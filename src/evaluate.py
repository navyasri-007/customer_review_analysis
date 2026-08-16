import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
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
    from src.data_preprocessing import prepare_dataset, load_reviews, get_dataset_file
except ImportError:
    from data_preprocessing import prepare_dataset, load_reviews, get_dataset_file


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "deberta",
    "final"
)

DEFAULT_FALLBACK_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# EVALUATION ENGINE
# ============================================================

def evaluate_model(sample_size=200, batch_size=32, model_path=None, save_plots=True):
    """
    Evaluates the sentiment model on the test dataset using batched inference.
    Computes accuracy, precision, recall, F1, and saves metrics & confusion matrix plot.
    """
    print("=" * 60)
    print("CUSTOMER REVIEW SENTIMENT MODEL EVALUATION")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device : {device}")

    target_model_path = model_path or MODEL_PATH
    has_custom_model = (
        os.path.isdir(target_model_path) and
        "config.json" in os.listdir(target_model_path) and
        any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(target_model_path))
    )

    # 1. Load Test Dataset
    print(f"\nLoading test reviews (target sample size: {sample_size:,})...")
    test_file = get_dataset_file("test.ft.txt")

    if os.path.isfile(test_file):
        test_df = load_reviews(max_per_class=sample_size // 2, source_file=test_file)
    else:
        _, _, test_df = prepare_dataset(max_per_class=sample_size)

    test_reviews = test_df["review"].tolist()
    actual_labels = test_df["label"].tolist()

    print(f"Total test examples loaded: {len(test_reviews):,}")
    print(f"  -> Positive (1): {actual_labels.count(1):,}")
    print(f"  -> Negative (0): {actual_labels.count(0):,}")

    predictions = []

    # 2. Load Model & Tokenizer
    if has_custom_model:
        model_name_or_path = target_model_path
        model_evaluated = f"Custom DeBERTa-v3 ({target_model_path})"
        print(f"\nLoading custom fine-tuned model from:\n  {target_model_path}")
    else:
        model_name_or_path = DEFAULT_FALLBACK_MODEL
        model_evaluated = f"Pretrained Transformer ({DEFAULT_FALLBACK_MODEL})"
        print(f"\nLocal model not found at '{target_model_path}'.")
        print(f"Using pretrained model: '{DEFAULT_FALLBACK_MODEL}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    # 3. Batched Inference
    print(f"\nEvaluating {len(test_reviews)} reviews in batches of {batch_size}...")
    for i in range(0, len(test_reviews), batch_size):
        batch_texts = test_reviews[i:i + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()

        predictions.extend(preds.tolist())

        progress = min(i + len(batch_texts), len(test_reviews))
        print(f"  Processed {progress}/{len(test_reviews)} reviews ({progress / len(test_reviews) * 100:.1f}%)")

    # 4. Compute Metrics
    accuracy = float(accuracy_score(actual_labels, predictions))
    precision = float(precision_score(actual_labels, predictions, average="binary", zero_division=0))
    recall = float(recall_score(actual_labels, predictions, average="binary", zero_division=0))
    f1 = float(f1_score(actual_labels, predictions, average="binary", zero_division=0))
    cm = confusion_matrix(actual_labels, predictions).tolist()

    report_dict = classification_report(
        actual_labels,
        predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )

    report_text = classification_report(
        actual_labels,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model Evaluated : {model_evaluated}")
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Precision       : {precision * 100:.2f}%")
    print(f"Recall          : {recall * 100:.2f}%")
    print(f"F1 Score        : {f1 * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report_text)
    print("Confusion Matrix:")
    print(f"  TN: {cm[0][0]:<6} | FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]:<6} | TP: {cm[1][1]}")

    # 5. Save Metrics to JSON
    metrics_summary = {
        "model_evaluated": model_evaluated,
        "sample_size": len(test_reviews),
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": {
            "true_negative": cm[0][0],
            "false_positive": cm[0][1],
            "false_negative": cm[1][0],
            "true_positive": cm[1][1]
        },
        "classification_report": report_dict
    }

    metrics_file = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved evaluation metrics to:\n  {metrics_file}")

    # 6. Plot & Save Confusion Matrix
    if save_plots:
        try:
            plt.figure(figsize=(6, 5), dpi=150)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Negative (0)", "Positive (1)"],
                yticklabels=["Negative (0)", "Positive (1)"],
                cbar=False
            )
            plt.title(f"Confusion Matrix\n(Accuracy: {accuracy*100:.1f}%, F1: {f1*100:.1f}%)", fontsize=12, fontweight="bold")
            plt.xlabel("Predicted Label", fontsize=10)
            plt.ylabel("True Label", fontsize=10)
            plt.tight_layout()

            plot_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved confusion matrix plot to:\n  {plot_path}")
        except Exception as e:
            print(f"Note: Could not generate confusion matrix plot ({e})")

    return metrics_summary


# ============================================================
# CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Customer Review Sentiment Model")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of test reviews to evaluate (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size (default: 32)")
    parser.add_argument("--model_path", type=str, default=None, help="Custom model path (default: models/deberta/final)")
    parser.add_argument("--no_plot", action="store_true", help="Disable confusion matrix plot generation")

    args = parser.parse_args()

    evaluate_model(
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        model_path=args.model_path,
        save_plots=not args.no_plot
    )