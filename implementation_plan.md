# Customer Review AI Agent - Operationalization & Dashboard Implementation Plan

## Problem & Context
The `customer_review` repository contains a Transformer-based NLP sentiment analysis and business action recommendation system. A comprehensive review of the codebase identified the following issues preventing execution:

1. **Dataset Path Resolution Bug**: In [data_preprocessing.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/data_preprocessing.py), dataset paths point directly to `train.ft.txt` and `test.ft.txt`, which in this environment are directories containing the actual text files. This causes an immediate `PermissionError` / path failure upon reading.
2. **Missing Dashboard Implementation**: [dashboard.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/dashboard.py) is currently empty (0 bytes), preventing the user from running `streamlit run dashboard.py`.
3. **Model Path & Fallback Handling**: [predict.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/predict.py) and [evaluate.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/evaluate.py) expect pre-existing local model weights at `models/deberta/final`. Before full model training completes, prediction and dashboard scripts would fail.
4. **Evaluation Inefficiency**: [evaluate.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/evaluate.py) iterates row-by-row on test samples instead of using batched evaluation, causing severe slowdowns.
5. **Import & Module Resolution**: Running scripts from different working directories (project root vs. `src/`) can cause `ModuleNotFoundError` without proper path initialization.

---

## Proposed Changes

### 1. Data Preprocessing Component
#### [MODIFY] [data_preprocessing.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/data_preprocessing.py)
- Implement `get_dataset_file(filename)` to resolve both nested directory structures (`data/amazonreviews/train.ft.txt/train.ft.txt`) and standard file paths.
- Add configurable sample loading and streaming support for fast verification and custom dataset splits.
- Save preprocessed metadata/stats to `results/dataset_summary.json`.

---

### 2. Business Analysis Component
#### [MODIFY] [business_analysis.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/business_analysis.py)
- Enhance priority scoring with confidence thresholds and sentiment signals.
- Add automated issue category tagger (Product Quality, Shipping & Delivery, Customer Service, Pricing & Billing, Defect/Usability).
- Add SLA response timeline calculation (e.g., High Priority: < 2 hours; Medium: 24 hours; Low: Routine review).
- Add comprehensive batch aggregation and exportable summary generation.

---

### 3. Prediction Pipeline
#### [MODIFY] [predict.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/predict.py)
- Implement graceful dual-mode loading:
  - If fine-tuned weights exist at `models/deberta/final`, load the custom DeBERTa-v3 model.
  - If fine-tuned weights are not yet created, fall back to a pretrained transformer sentiment pipeline (`distilbert-base-uncased-finetuned-sst-2-english` or `microsoft/deberta-v3-base`) so all predictions and the dashboard work immediately out-of-the-box.
- Add batched prediction `predict_batch()` for rapid multi-review analysis.
- Include probability distribution breakdown (Positive % vs. Negative %).

---

### 4. Training & Evaluation Pipeline
#### [MODIFY] [train.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/train.py)
- Add CLI argument support (`--epochs`, `--batch_size`, `--sample_size`, `--learning_rate`, `--quick_run`) to allow quick testing or full training.
- Ensure device management (CUDA / CPU), proper metric logging, and persistence of training metrics to `results/training_metrics.json`.

#### [MODIFY] [evaluate.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/src/evaluate.py)
- Implement batched inference for test evaluation (10x-20x speedup).
- Calculate Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and full Classification Report.
- Save evaluation metrics to `results/evaluation_metrics.json` and generate confusion matrix plot in `results/confusion_matrix.png`.

---

### 5. Streamlit Dashboard
#### [MODIFY] [dashboard.py](file:///c:/Users/navya/OneDrive/Documents/customer_review/dashboard.py)
Create a comprehensive, modern Streamlit dashboard featuring:
- **Live Sentiment Analyzer**: Real-time review classification with confidence gauge, priority badge, SLA turnaround timer, and recommended business actions.
- **Batch Processor & CSV Uploader**: Upload customer review CSV/Excel files, run batch inference, preview interactive data tables, and download annotated reports.
- **Executive BI & Analytics Overview**: KPI metric tiles, sentiment breakdown charts, priority distribution, confidence histograms, and urgency escalation feed.
- **Dataset & Model Explorer**: Explore dataset statistics, label distribution, DeBERTa architecture specifications, and evaluation metrics.
- **Polished Modern UI**: Custom CSS styling with cards, badges, and responsive tabs.

---

## Verification Plan

### Automated Verification
1. **Data Preprocessing Test**: Run `python src/data_preprocessing.py` to confirm successful loading, path resolution, and balanced splitting.
2. **Business Analysis Test**: Run `python src/business_analysis.py` to verify priority calculation and summary aggregation.
3. **Prediction Test**: Run `python src/predict.py` to confirm single and batch inference with business action enrichment.
4. **Evaluation Test**: Run `python src/evaluate.py --sample_size 100` to verify metric computation and results export.
5. **Dashboard Launch Test**: Start Streamlit in headless mode or verify syntax and startup without runtime errors.
