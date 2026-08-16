import os
import sys
import json
import time

# Suppress Hugging Face & tokenizer warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# PROJECT & MODULE PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.predict import SentimentPredictor, get_predictor
from src.business_analysis import (
    analyze_prediction,
    detect_issue_category,
    calculate_priority,
    get_sla_turnaround,
    generate_recommended_action,
    create_business_summary
)
from src.data_preprocessing import get_dataset_file

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Review AI Agent | Sentiment & Action Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Result Badges */
    .badge-pos {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid #22c55e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-neg {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-med {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid #3b82f6;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    /* Action Card */
    .action-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border-left: 4px solid #818cf8;
        border-radius: 8px;
        padding: 16px;
        margin-top: 15px;
    }
    
    /* Custom divider */
    .custom-hr {
        border: 0;
        height: 1px;
        background: #334155;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED PREDICTOR LOADER (FAST IN-MEMORY SINGLETON)
# ============================================================

@st.cache_resource(show_spinner="⚡ Loading Transformer Sentiment Model...")
def load_sentiment_model():
    return get_predictor()


# ============================================================
# PRESET SAMPLE REVIEWS
# ============================================================

SAMPLE_REVIEWS = {
    "⚡ Critical Hardware Defect (High Priority)": (
        "The blender motor caught fire on day two! Terrible manufacturing quality and dangerous defect. "
        "I demand an immediate replacement or full refund before I file a safety complaint."
    ),
    "📦 Lost Shipping & Delayed Delivery (High Priority)": (
        "My order never arrived. The tracking number has shown 'In Transit' for 3 weeks with no updates. "
        "Customer service has not replied to my emails. Completely unacceptable shipping delay."
    ),
    "💬 Rude Customer Support (High Priority)": (
        "I reached out for a simple warranty claim and the agent was dismissive, rude, and hung up on me. "
        "Worst customer support experience ever."
    ),
    "⭐ Glowing 5-Star Product Review (Low Priority / Praise)": (
        "This mechanical keyboard is absolutely phenomenal! The key switches feel smooth, the build is sturdy metal, "
        "and typing on it is pure joy. Best purchase I have made all year. Highly recommended!"
    ),
    "💰 Pricing & Value Feedback (Medium/Low Priority)": (
        "The item works fine, but the price is definitely inflated for what you get. It feels a bit too lightweight "
        "and cheap considering the premium brand cost."
    ),
    "🛠️ Usability & Setup Issue (Medium Priority)": (
        "The software installation manual is confusing and incomplete. It took me 4 hours to get it running. "
        "Please provide clearer instructions or a video setup guide."
    )
}

BATCH_PRESETS = [
    "The camera lens cracked inside the box before I even opened it. Awful packaging!",
    "Super fast delivery, arrived 2 days earlier than scheduled. Package in pristine condition.",
    "The noise cancelling on these headphones is top tier. Crystal clear audio on my flights.",
    "Customer service rep was very helpful and processed my replacement within 5 minutes.",
    "Battery drained from 100% to 0% in under 45 minutes. Total waste of money.",
    "Software crashed repeatedly on Windows 11. Support has not provided a bug fix yet.",
    "Solid quality, beautiful design, well worth the price. Five stars!",
    "Wrong color shipped and missing USB-C charging cable from the box. Please fix this.",
    "Decent product for casual use, though buttons feel somewhat loose.",
    "Outstanding durability! Dropped it twice on concrete and no scratches at all."
]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/artificial-intelligence.png", width=64)
    st.title("Customer Review AI")
    st.caption("DeBERTa-v3 NLP & Action Agent")

    st.markdown("---")
    navigation = st.radio(
        "Navigation",
        [
            "🎯 Live Review Analyzer",
            "📁 Batch CSV & Bulk Analysis",
            "📊 Executive Business Intelligence",
            "🔬 Dataset & Model Inspector"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("⚙️ System Status")

    # Load Model status
    try:
        predictor = load_sentiment_model()
        model_status = f"✅ {predictor.model_name}"
        device_status = f"⚡ {predictor.device.type.upper()}"
    except Exception as e:
        model_status = f"⚠️ Fallback ({e})"
        device_status = "CPU"

    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Compute:** {device_status}")

    # Dataset status
    train_file = get_dataset_file("train.ft.txt")
    if os.path.isfile(train_file):
        st.markdown(f"**Dataset:** ✅ Connected ({os.path.basename(train_file)})")
    else:
        st.markdown("**Dataset:** ⚠️ Files Missing")

    st.markdown("---")
    st.info("💡 **Tip:** Model classifies sentiment, identifies root-cause issues, calculates SLA response time, and provides actionable business steps.")


# ============================================================
# VIEW 1: LIVE REVIEW ANALYZER
# ============================================================

if navigation == "🎯 Live Review Analyzer":
    st.title("🎯 Real-Time Review Sentiment & Action Analyzer")
    st.markdown("Enter any customer review to instantly predict sentiment, diagnose root cause category, and generate automated business actions.")

    col_input, col_preset = st.columns([2, 1])

    with col_preset:
        st.markdown("#### ⚡ Quick Presets")
        preset_choice = st.selectbox(
            "Select an example review:",
            ["Custom Input..."] + list(SAMPLE_REVIEWS.keys()),
            index=0
        )
        selected_text = ""
        if preset_choice != "Custom Input...":
            selected_text = SAMPLE_REVIEWS[preset_choice]

    with col_input:
        st.markdown("#### 📝 Customer Review Text")
        user_review = st.text_area(
            "Review Content",
            value=selected_text,
            height=140,
            placeholder="Type or paste a customer review here (e.g. 'The product stopped working after two weeks...')"
        )

        analyze_btn = st.button("🚀 Analyze Review & Generate Action", type="primary", use_container_width=True)

    if analyze_btn or (user_review and preset_choice != "Custom Input..."):
        if not user_review.strip():
            st.warning("Please enter some review text to analyze.")
        else:
            predictor = load_sentiment_model()
            result = predictor.predict_and_analyze(user_review)

            st.markdown("<hr class='custom-hr'/>", unsafe_allow_html=True)
            st.subheader("📋 AI Agent Analysis Results")

            # Top KPI metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)

            with col_m1:
                sentiment = result["Sentiment"]
                badge_class = "badge-pos" if sentiment == "Positive" else "badge-neg"
                icon = "🟢" if sentiment == "Positive" else "🔴"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Sentiment</div>
                    <div class="metric-val"><span class="{badge_class}">{icon} {sentiment}</span></div>
                    <div class="metric-sub">Confidence: {result['Confidence']}%</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m2:
                priority = result["Priority"]
                if priority == "High":
                    p_badge = "badge-high"
                elif priority == "Medium":
                    p_badge = "badge-med"
                else:
                    p_badge = "badge-low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Operational Priority</div>
                    <div class="metric-val"><span class="{p_badge}">{priority} Priority</span></div>
                    <div class="metric-sub">Urgency scoring</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Root Cause Category</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0; margin-top: 6px;">
                        📌 {result['Category']}
                    </div>
                    <div class="metric-sub">NLP issue classifier</div>
                </div>
                """, unsafe_allow_html=True)

            with col_m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Target SLA Response</div>
                    <div style="font-size: 1.1rem; font-weight: 600; color: #38bdf8; margin-top: 6px;">
                        ⏱️ {result['SLA_Target']}
                    </div>
                    <div class="metric-sub">Escalation window</div>
                </div>
                """, unsafe_allow_html=True)

            # Recommended Action Box
            st.markdown(f"""
            <div class="action-box">
                <div style="font-size: 1.05rem; font-weight: 700; color: #a5b4fc; margin-bottom: 6px;">
                    🎯 Recommended Business Action:
                </div>
                <div style="font-size: 1.15rem; color: #f8fafc; font-weight: 500;">
                    {result['Recommended_Action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detailed Probability Bars
            st.markdown("#### 📊 Sentiment Probability Breakdown")
            col_pb1, col_pb2 = st.columns(2)
            with col_pb1:
                st.write(f"**Positive Sentiment Probability:** {result.get('pos_prob', 0)}%")
                st.progress(float(result.get('pos_prob', 0)) / 100.0)
            with col_pb2:
                st.write(f"**Negative Sentiment Probability:** {result.get('neg_prob', 0)}%")
                st.progress(float(result.get('neg_prob', 0)) / 100.0)


# ============================================================
# VIEW 2: BATCH CSV & BULK ANALYSIS
# ============================================================

elif navigation == "📁 Batch CSV & Bulk Analysis":
    st.title("📁 Batch Reviews & CSV Analysis")
    st.markdown("Process hundreds of customer reviews simultaneously. Upload a CSV file or analyze our instant demo batch.")

    tab_upload, tab_sample = st.tabs(["📤 Upload CSV / Excel File", "⚡ Run Demo Batch (10 Reviews)"])

    df_to_process = None
    text_column_name = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file containing customer reviews",
            type=["csv", "xlsx", "txt"]
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_to_process = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df_to_process = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".txt"):
                    lines = [line.decode("utf-8", errors="ignore").strip() for line in uploaded_file.readlines()]
                    lines = [l for l in lines if l]
                    df_to_process = pd.DataFrame({"review": lines})

                st.success(f"Loaded {len(df_to_process):,} rows from '{uploaded_file.name}'")
                candidate_cols = [c for c in df_to_process.columns if any(k in c.lower() for k in ["review", "text", "comment", "feedback", "body"])]
                default_idx = df_to_process.columns.get_loc(candidate_cols[0]) if candidate_cols else 0
                text_column_name = st.selectbox("Select column containing review text:", df_to_process.columns, index=default_idx)
            except Exception as e:
                st.error(f"Error parsing file: {e}")

    with tab_sample:
        st.markdown("Load pre-configured real-world customer reviews across shipping, defects, praise, and billing:")
        if st.button("Load Demo Batch Dataset", type="secondary"):
            df_to_process = pd.DataFrame({"review": BATCH_PRESETS})
            text_column_name = "review"
            st.session_state["demo_batch_df"] = df_to_process

        if "demo_batch_df" in st.session_state and df_to_process is None:
            df_to_process = st.session_state["demo_batch_df"]
            text_column_name = "review"

    if df_to_process is not None and text_column_name is not None:
        st.markdown("---")
        st.subheader(f"📊 Dataset Preview ({len(df_to_process)} entries)")
        st.dataframe(df_to_process.head(5), use_container_width=True)

        batch_size_choice = st.slider("Select maximum rows to analyze:", 5, min(1000, len(df_to_process)), min(50, len(df_to_process)))

        if st.button("🚀 Run Fast Batch Sentiment & Action Engine", type="primary"):
            subset_df = df_to_process.head(batch_size_choice).copy()
            predictor = load_sentiment_model()

            reviews_list = subset_df[text_column_name].astype(str).tolist()

            with st.spinner(f"⚡ Processing {len(reviews_list)} reviews with vectorized transformer inference..."):
                start_time = time.time()
                # Fast vectorized tensor batching
                enriched_records = predictor.predict_and_analyze_batch(reviews_list, batch_size=32)
                elapsed = time.time() - start_time

            st.success(f"✅ Processed {len(reviews_list)} reviews in {elapsed:.2f}s ({len(reviews_list)/max(0.001, elapsed):.1f} reviews/sec)!")

            results_df = pd.DataFrame(enriched_records)
            st.session_state["last_batch_results"] = results_df

    if "last_batch_results" in st.session_state:
        res_df = st.session_state["last_batch_results"]

        st.markdown("<hr class='custom-hr'/>", unsafe_allow_html=True)
        st.subheader("📋 Enriched Review Analysis & Action Table")

        # Filters
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            sentiment_filter = st.multiselect("Filter by Sentiment:", options=["Positive", "Negative"], default=["Positive", "Negative"])
        with f_col2:
            priority_filter = st.multiselect("Filter by Priority:", options=["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        with f_col3:
            all_cats = list(res_df["Category"].unique())
            category_filter = st.multiselect("Filter by Category:", options=all_cats, default=all_cats)

        filtered_df = res_df[
            (res_df["Sentiment"].isin(sentiment_filter)) &
            (res_df["Priority"].isin(priority_filter)) &
            (res_df["Category"].isin(category_filter))
        ]

        st.dataframe(filtered_df, use_container_width=True)

        # Download CSV
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Annotated CSV Report",
            data=csv_data,
            file_name="customer_reviews_analyzed_with_actions.csv",
            mime="text/csv",
            type="primary"
        )


# ============================================================
# VIEW 3: EXECUTIVE BUSINESS INTELLIGENCE
# ============================================================

elif navigation == "📊 Executive Business Intelligence":
    st.title("📊 Executive Business Intelligence & KPI Dashboard")
    st.markdown("Actionable analytics, customer friction points, sentiment distributions, and urgent escalation tracking.")

    if "last_batch_results" in st.session_state:
        df_analytics = st.session_state["last_batch_results"]
    else:
        predictor = load_sentiment_model()
        preset_enriched = predictor.predict_and_analyze_batch(BATCH_PRESETS, batch_size=32)
        df_analytics = pd.DataFrame(preset_enriched)

    summary = create_business_summary(df_analytics.to_dict(orient="records"))

    # Top KPI Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Reviews Analyzed</div>
            <div class="metric-val">{summary['Total Reviews']}</div>
            <div class="metric-sub">Customer Volume</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Positive Sentiment</div>
            <div class="metric-val" style="color: #22c55e;">{summary['Positive Percentage']}%</div>
            <div class="metric-sub">{summary['Positive Reviews']} positive reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Negative Sentiment</div>
            <div class="metric-val" style="color: #ef4444;">{summary['Negative Percentage']}%</div>
            <div class="metric-sub">{summary['Negative Reviews']} complaints/issues</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High Priority Alerts</div>
            <div class="metric-val" style="color: #f87171;">{summary['High Priority Reviews']}</div>
            <div class="metric-sub">Urgent response (&lt; 2h)</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Model Confidence</div>
            <div class="metric-val" style="color: #a855f7;">{summary['Average Confidence']}%</div>
            <div class="metric-sub">Classification certainty</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='custom-hr'/>", unsafe_allow_html=True)

    # Charts Row
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        st.subheader("🍩 Sentiment Distribution")
        sent_counts = df_analytics["Sentiment"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        st.bar_chart(sent_counts.set_index("Sentiment"), color="#38bdf8")

    with ch_col2:
        st.subheader("🏷️ Issue Category Breakdown")
        cat_counts = df_analytics["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        st.bar_chart(cat_counts.set_index("Category"), color="#818cf8")

    # Urgent Escalation Feed
    st.markdown("<hr class='custom-hr'/>", unsafe_allow_html=True)
    st.subheader("🚨 High Priority Escalation Queue")
    st.caption("Reviews flagged as urgent requiring customer service intervention.")

    high_priority_df = df_analytics[df_analytics["Priority"] == "High"]
    if len(high_priority_df) > 0:
        st.dataframe(
            high_priority_df[["Review", "Category", "Confidence", "SLA_Target", "Recommended_Action"]],
            use_container_width=True
        )
    else:
        st.success("🎉 No High Priority escalation tickets currently open!")


# ============================================================
# VIEW 4: DATASET & MODEL INSPECTOR
# ============================================================

elif navigation == "🔬 Dataset & Model Inspector":
    st.title("🔬 Dataset & Transformer Architecture Inspector")
    st.markdown("Examine the Amazon Customer Reviews training corpus, DeBERTa-v3 architecture, and saved evaluation benchmarks.")

    tab_eval, tab_model, tab_data = st.tabs(["📈 Evaluation Benchmarks", "🤖 Model Architecture", "📦 Dataset Inspector"])

    with tab_eval:
        st.subheader("Evaluation Performance Metrics")
        metrics_file = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
        cm_plot_file = os.path.join(RESULTS_DIR, "confusion_matrix.png")

        if os.path.exists(metrics_file):
            with open(metrics_file, "r", encoding="utf-8") as f:
                saved_metrics = json.load(f)

            ec1, ec2, ec3, ec4 = st.columns(4)
            with ec1:
                st.metric("Test Accuracy", f"{saved_metrics.get('accuracy', 0)}%")
            with ec2:
                st.metric("Precision", f"{saved_metrics.get('precision', 0)}%")
            with ec3:
                st.metric("Recall", f"{saved_metrics.get('recall', 0)}%")
            with ec4:
                st.metric("F1 Score", f"{saved_metrics.get('f1_score', 0)}%")

            if os.path.exists(cm_plot_file):
                st.image(cm_plot_file, caption="Confusion Matrix on Test Dataset", width=450)

            st.json(saved_metrics)
        else:
            st.info("No saved evaluation metrics found yet in `results/evaluation_metrics.json`.")

    with tab_model:
        st.subheader("DeBERTa-v3 Model Specifications")
        st.markdown("""
        - **Model Checkpoint**: `microsoft/deberta-v3-base`
        - **Architecture**: Disentangled Attention with Enhanced Masked Language Modeling
        - **Layers**: 12 Transformer Encoder Blocks
        - **Hidden Dimension**: 768
        - **Attention Heads**: 12
        - **Vocabulary Size**: 128,100 (SentencePiece SPM)
        - **Classification Head**: Binary Sequence Classification (Negative: 0, Positive: 1)
        - **Optimizer & Scheduler**: AdamW (`lr=2e-5`, `weight_decay=0.01`, Linear Warmup)
        """)

    with tab_data:
        st.subheader("Amazon Customer Reviews Dataset")
        st.markdown("""
        - **Source Corpus**: FastText Amazon Reviews Dataset
        - **Working Balanced Split**:
          - **Training**: 14,000 balanced reviews (7,000 Positive, 7,000 Negative)
          - **Validation**: 3,000 balanced reviews (1,500 Positive, 1,500 Negative)
          - **Test**: 3,000 balanced reviews (1,500 Positive, 1,500 Negative)
        - **Label Mapping**:
          - `__label__1` $\\rightarrow$ Negative (0)
          - `__label__2` $\\rightarrow$ Positive (1)
        """)

        train_path = get_dataset_file("train.ft.txt")
        st.write(f"**Data File Path:** `{train_path}`")
        if os.path.isfile(train_path):
            st.write(f"**File Size:** {os.path.getsize(train_path) / (1024*1024):.2f} MB")
