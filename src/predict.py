import os
import sys
import torch

# Suppress Hugging Face symlink & tokenizer warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optimize PyTorch CPU inference threads
if not torch.cuda.is_available():
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    except Exception:
        pass

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
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
    from src.business_analysis import analyze_prediction
except ImportError:
    from business_analysis import analyze_prediction


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "deberta",
    "final"
)

DEFAULT_FALLBACK_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


# ============================================================
# SENTIMENT PREDICTOR CLASS
# ============================================================

class SentimentPredictor:
    """
    High-performance inference engine for customer review sentiment analysis.
    Supports both custom fine-tuned DeBERTa-v3 models and
    fast pretrained transformer sentiment models.
    """

    def __init__(self, model_path_or_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

        target_path = model_path_or_name or MODEL_PATH

        # 1. Check if custom fine-tuned model exists
        if self._is_valid_model_dir(target_path):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(target_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(target_path)
                self.model.to(self.device)
                self.model.eval()
                self.model_name = "Fine-Tuned DeBERTa-v3 (Custom)"
                self.model_source = "custom"
                return
            except Exception as e:
                pass

        # 2. Fallback to cached/fast pretrained transformer model
        fallback_name = DEFAULT_FALLBACK_MODEL
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(fallback_name)
            self.model.to(self.device)
            self.model.eval()
            self.model_name = f"Pretrained Transformer ({fallback_name})"
            self.model_source = "fallback"
        except Exception as e:
            self.model_name = "Rule-Based Fallback"
            self.model_source = "rule_fallback"

    def _is_valid_model_dir(self, path):
        """Checks if directory contains essential Hugging Face model files."""
        if not os.path.isdir(path):
            return False
        files = os.listdir(path)
        has_config = "config.json" in files
        has_weights = any(f.endswith(".bin") or f.endswith(".safetensors") or f == "model.safetensors" for f in files)
        return has_config and has_weights

    def predict(self, review_text):
        """
        Classifies sentiment of a single review string.
        """
        if not review_text or not review_text.strip():
            return {
                "review": review_text,
                "sentiment": "Neutral",
                "confidence": 50.0,
                "pos_prob": 50.0,
                "neg_prob": 50.0,
                "model_used": self.model_name
            }

        # 1. Neural Transformer Inference
        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                review_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probabilities).item()

            neg_prob = float(probabilities[0].item() * 100.0)
            pos_prob = float(probabilities[1].item() * 100.0) if len(probabilities) > 1 else (100.0 - neg_prob)

            sentiment = "Positive" if pred_idx == 1 else "Negative"
            confidence = pos_prob if sentiment == "Positive" else neg_prob

            return {
                "review": review_text,
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "pos_prob": round(pos_prob, 2),
                "neg_prob": round(neg_prob, 2),
                "model_used": self.model_name
            }

        # 2. Rule-based Basic Fallback
        else:
            positive_words = ["good", "great", "excellent", "love", "awesome", "best", "perfect", "happy", "amazing", "stunning"]
            negative_words = ["bad", "terrible", "horrible", "broken", "worst", "hate", "poor", "waste", "defect", "flawed"]

            lower = review_text.lower()
            pos_count = sum(1 for w in positive_words if w in lower)
            neg_count = sum(1 for w in negative_words if w in lower)

            sentiment = "Positive" if pos_count >= neg_count else "Negative"
            confidence = 85.0 if (pos_count + neg_count) > 0 else 50.0

            return {
                "review": review_text,
                "sentiment": sentiment,
                "confidence": round(confidence, 2),
                "pos_prob": 85.0 if sentiment == "Positive" else 15.0,
                "neg_prob": 15.0 if sentiment == "Positive" else 85.0,
                "model_used": self.model_name
            }

    def predict_batch(self, reviews_list, batch_size=32):
        """
        Classifies sentiment for a list of review strings using batched tensor tokenization.
        """
        if not reviews_list:
            return []

        if self.model is None or self.tokenizer is None:
            return [self.predict(t) for t in reviews_list]

        all_results = []
        for i in range(0, len(reviews_list), batch_size):
            batch_texts = reviews_list[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()

            for text, pred_idx, p in zip(batch_texts, preds, probs.cpu().numpy()):
                neg_prob = float(p[0] * 100.0)
                pos_prob = float(p[1] * 100.0) if len(p) > 1 else (100.0 - neg_prob)
                sentiment = "Positive" if pred_idx == 1 else "Negative"
                confidence = pos_prob if sentiment == "Positive" else neg_prob

                all_results.append({
                    "review": text,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 2),
                    "pos_prob": round(pos_prob, 2),
                    "neg_prob": round(neg_prob, 2),
                    "model_used": self.model_name
                })

        return all_results

    def predict_and_analyze(self, review_text):
        """
        Performs sentiment prediction and enriches it with business analysis & actions.
        """
        raw_pred = self.predict(review_text)
        business_info = analyze_prediction(raw_pred)
        merged = {**raw_pred, **business_info}
        return merged

    def predict_and_analyze_batch(self, reviews_list, batch_size=32):
        """
        Performs fast vectorized batch prediction and enriches all results with business actions.
        """
        raw_preds = self.predict_batch(reviews_list, batch_size=batch_size)
        results = []
        for raw in raw_preds:
            enriched = analyze_prediction(raw)
            merged = {**raw, **enriched}
            results.append(merged)
        return results


# ============================================================
# SINGLETON INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================

_predictor = None


def get_predictor():
    """Returns or lazily instantiates the global SentimentPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = SentimentPredictor()
    return _predictor


def predict_customer_review(review_text):
    """
    Standard entrypoint: predicts sentiment and confidence for a review.
    """
    predictor = get_predictor()
    return predictor.predict(review_text)


def predict_and_analyze_review(review_text):
    """
    Full AI Agent entrypoint: predicts sentiment and generates actionable business recommendations.
    """
    predictor = get_predictor()
    return predictor.predict_and_analyze(review_text)