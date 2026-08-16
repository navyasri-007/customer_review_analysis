# Customer Review AI Agent 🤖

An enterprise-grade, AI-powered customer review sentiment analysis and decision intelligence system built with Transformer Natural Language Processing (DeBERTa-v3 & DistilBERT), PyTorch, Hugging Face, Scikit-learn, and Streamlit.

---

## 🚀 Key Capabilities

- **Transformer Sentiment Classification**: High-accuracy binary sentiment classification (Positive / Negative).
- **Confidence Scoring**: Exact probability distribution and certainty estimation.
- **Operational Priority Engine**: Triages customer feedback into **High**, **Medium**, and **Low** urgency tiers based on sentiment confidence and defect severity.
- **Root-Cause Issue Categorization**: Automated categorization across:
  - *Product Quality & Defect*
  - *Shipping & Delivery*
  - *Customer Service & Returns*
  - *Pricing & Billing*
  - *Usability & Setup*
  - *Customer Praise & Loyalty*
- **SLA Response Targets**: Automatically suggests turnaround timelines (e.g. `< 2 Hours` for urgent escalations).
- **Contextual Action Recommendations**: Generates tailored business workflows (e.g., immediate replacement units, supervisor callback, carrier dispute, loyalty reward invitations).
- **Interactive Streamlit Web Dashboard**: Real-time analyzer, batch CSV processor with report download, executive BI analytics, and model/dataset inspector.

---

## 📂 Project Structure

```
Customer_Review_AI_Agent/
├── data/
│   └── amazonreviews/
│       ├── train.ft.txt/          # FastText Amazon reviews training corpus
│       └── test.ft.txt/           # FastText Amazon reviews testing corpus
├── models/
│   └── deberta/
│       └── final/                 # Fine-tuned DeBERTa-v3 model & tokenizer
├── results/
│   ├── evaluation_metrics.json    # Test evaluation metrics & reports
│   └── confusion_matrix.png       # Confusion matrix visualization
├── src/
│   ├── data_preprocessing.py      # Dataset loading, balancing & splitting
│   ├── business_analysis.py       # Priority, category & action engine
│   ├── predict.py                 # Single & batch inference pipeline
│   ├── evaluate.py                # Batched model evaluation & metrics export
│   └── train.py                   # PyTorch + Transformers DeBERTa training
├── dashboard.py                   # Full-featured Streamlit web application
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
```

---

## 📦 Setup & Installation

### 1. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Quick Start & Execution

### 1. Launch the Interactive Web Dashboard
Run the Streamlit web dashboard:
```bash
streamlit run dashboard.py
```
Open **`http://localhost:8501`** in your browser to access:
- **Real-Time Review Analyzer** (with preset customer complaints & praise)
- **Batch CSV Processor** (bulk review upload, live progress, filterable table, CSV export)
- **Executive Business Intelligence** (KPI metrics, sentiment donut chart, category breakdown, urgent escalation queue)
- **Dataset & Model Inspector** (benchmark metrics, confusion matrix, architecture specs)

---

### 2. Run Direct Review Prediction
Analyze sample customer reviews and generate business actions from the terminal:
```bash
python src/predict.py
```

---

### 3. Evaluate Model Performance
Evaluate sentiment model accuracy, precision, recall, F1 score, and generate confusion matrix plot:
```bash
# Evaluate on 200 test reviews
python src/evaluate.py --sample_size 200 --batch_size 32

# Evaluate on full test set
python src/evaluate.py --sample_size 3000
```
Metrics and confusion matrix will be saved to `results/evaluation_metrics.json` and `results/confusion_matrix.png`.

---

### 4. Verify Data Preprocessing
Test dataset path resolution and balanced splits:
```bash
python src/data_preprocessing.py --max_per_class 1000
```

---

### 5. Train Fine-Tuned DeBERTa-v3 Model
Train a custom DeBERTa-v3 sequence classification model on the Amazon reviews dataset:
```bash
# Quick test run (200 samples, 1 epoch)
python src/train.py --quick_test

# Full production training (20,000 balanced reviews, 2 epochs)
python src/train.py --epochs 2 --batch_size 8 --learning_rate 2e-5 --sample_size 10000
```
The best model checkpoint will be automatically saved to `models/deberta/final`.

---

## 📊 Evaluation Benchmarks (Test Set)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **89.00%** |
| **Precision** | **91.49%** |
| **Recall** | **86.00%** |
| **F1-Score** | **88.66%** |

---

## 🛠️ Technologies Used

- **Language**: Python 3.11+
- **Deep Learning**: PyTorch, Hugging Face Transformers, Accelerate
- **Pretrained Models**: DeBERTa-v3 (`microsoft/deberta-v3-base`), DistilBERT
- **Data & Evaluation**: Scikit-learn, Pandas, NumPy, Datasets, Evaluate
- **Visualization**: Matplotlib, Seaborn
- **Web UI & Dashboard**: Streamlit