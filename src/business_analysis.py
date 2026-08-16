import os
import sys
import re
import pandas as pd

# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# ISSUE CATEGORY CLASSIFIER
# ============================================================

CATEGORY_KEYWORDS = {
    "Product Quality & Defect": [
        r"\bbroke\b", r"\bbroken\b", r"\bdefective\b", r"\bdefect\b",
        r"\bstopped working\b", r"\bfell apart\b", r"\bjunk\b", r"\btrash\b",
        r"\bterrible quality\b", r"\bpoor quality\b", r"\bflimsy\b",
        r"\bshoddy\b", r"\bcracked\b", r"\bmalfunction\b", r"\bburn\b",
        r"\bdamaged\b", r"\bcheaply made\b", r"\bfailed\b"
    ],
    "Shipping & Delivery": [
        r"\bship\b", r"\bshipping\b", r"\bdelivery\b", r"\bdelivered\b",
        r"\blate\b", r"\bdelay\b", r"\bdelayed\b", r"\bcourier\b",
        r"\bpackage\b", r"\bpackaging\b", r"\bnever arrived\b",
        r"\blost in transit\b", r"\btracking\b", r"\bbox crushed\b"
    ],
    "Customer Service & Returns": [
        r"\bcustomer service\b", r"\bsupport\b", r"\brefund\b", r"\breturn\b",
        r"\breturned\b", r"\breplacement\b", r"\bwarranty\b", r"\brepresentative\b",
        r"\brude\b", r"\bunresponsive\b", r"\bno reply\b", r"\brefused\b"
    ],
    "Pricing & Billing": [
        r"\boverpriced\b", r"\brip off\b", r"\bexpensive\b", r"\bcharge\b",
        r"\bcharged\b", r"\bbilling\b", r"\bhidden fee\b", r"\bnot worth\b",
        r"\bwaste of money\b", r"\bcost\b", r"\bscam\b"
    ],
    "Usability & Compatibility": [
        r"\bhard to use\b", r"\bdifficult\b", r"\bconfusing\b", r"\bmanual\b",
        r"\binstructions\b", r"\bnot as described\b", r"\bmisleading\b",
        r"\bwrong size\b", r"\bincompatible\b", r"\bdoesn't fit\b",
        r"\bsetup\b", r"\binstallation\b"
    ]
}


def detect_issue_category(review_text, sentiment="Negative"):
    """
    Detects root cause category based on text content and sentiment.
    """
    text_lower = review_text.lower()

    if sentiment == "Positive":
        # Check if it highlights specific praise
        if any(w in text_lower for w in ["fast shipping", "quick delivery", "arrived early"]):
            return "Fast Delivery & Logistics"
        if any(w in text_lower for w in ["support", "service", "helpful", "friendly"]):
            return "Exceptional Customer Support"
        if any(w in text_lower for w in ["great price", "good value", "affordable", "deal", "worth the money"]):
            return "High Value for Price"
        return "Product Quality & Satisfaction"

    # For Negative or Neutral reviews, check issue patterns
    matched_categories = []
    for category, patterns in CATEGORY_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                matched_categories.append(category)
                break

    if matched_categories:
        return matched_categories[0]

    return "General Experience & Feedback"


# ============================================================
# PRIORITY & SLA CALCULATION
# ============================================================

def calculate_priority(sentiment, confidence):
    """
    Calculates operational priority level based on sentiment and confidence.
    """
    if sentiment == "Negative":
        if confidence >= 85.0:
            return "High"
        elif confidence >= 65.0:
            return "Medium"
        else:
            return "Low"
    return "Low"


def get_sla_turnaround(priority):
    """
    Returns recommended SLA target response time.
    """
    if priority == "High":
        return "< 2 Hours (Urgent Escalation)"
    elif priority == "Medium":
        return "< 24 Hours (Next-Day Follow-Up)"
    else:
        return "3-5 Business Days (Standard Monitoring)"


# ============================================================
# ACTION RECOMMENDATION ENGINE
# ============================================================

def generate_recommended_action(sentiment, priority, category):
    """
    Generates specific, contextual business actions based on classification.
    """
    if sentiment == "Negative":
        if priority == "High":
            if category == "Product Quality & Defect":
                return "Immediate outreach: Offer free replacement unit or full refund; flag lot/batch for QA inspection."
            elif category == "Shipping & Delivery":
                return "Urgent logistics ticket: Verify tracking status, initiate carrier dispute, and expedite replacement shipment."
            elif category == "Customer Service & Returns":
                return "Supervisor escalation: Assign senior support specialist to contact customer directly within 2 hours."
            elif category == "Pricing & Billing":
                return "Billing audit: Review charge discrepancy and issue goodwill credit/adjustment."
            else:
                return "High-priority customer outreach: Proactively contact customer to resolve issue and prevent churn."

        elif priority == "Medium":
            if category == "Product Quality & Defect":
                return "Standard RMA review: Send troubleshooting guide and prepaid return label."
            elif category == "Shipping & Delivery":
                return "Check delivery milestone updates and send tracking notification to buyer."
            elif category == "Usability & Compatibility":
                return "Send product user guide, sizing chart, or quick-start video."
            else:
                return "Queue for customer success follow-up within 24 business hours."

        else:  # Low priority negative
            return "Log feedback in CRM for product analytics; monitor for recurring patterns."

    else:  # Positive sentiment
        if category == "High Value for Price" or category == "Product Quality & Satisfaction":
            return "Send thank-you email; invite customer to loyalty rewards or referral program."
        elif category == "Exceptional Customer Support":
            return "Acknowledge support agent mentioned in review; log customer delight metric."
        else:
            return "No immediate action required. Log positive sentiment in brand health dashboard."


# ============================================================
# SINGLE PREDICTION ENRICHMENT
# ============================================================

def analyze_prediction(result):
    """
    Enriches a raw sentiment prediction with business metrics and actions.
    
    Expected result format:
      {"review": str, "sentiment": "Positive"|"Negative", "confidence": float}
    """
    review = result.get("review", "")
    sentiment = result.get("sentiment", "Positive")
    confidence = float(result.get("confidence", 0.0))

    priority = calculate_priority(sentiment, confidence)
    category = detect_issue_category(review, sentiment)
    sla = get_sla_turnaround(priority)
    action = generate_recommended_action(sentiment, priority, category)

    return {
        "Review": review,
        "Sentiment": sentiment,
        "Confidence": round(confidence, 2),
        "Priority": priority,
        "Category": category,
        "SLA_Target": sla,
        "Recommended_Action": action
    }


# ============================================================
# BATCH SUMMARY & KPI GENERATION
# ============================================================

def create_business_summary(results):
    """
    Generates executive summary KPI metrics from a collection of analyzed reviews.
    """
    if not results:
        return {
            "Total Reviews": 0,
            "Positive Reviews": 0,
            "Negative Reviews": 0,
            "Positive Percentage": 0.0,
            "Negative Percentage": 0.0,
            "High Priority Reviews": 0,
            "Medium Priority Reviews": 0,
            "Low Priority Reviews": 0,
            "Average Confidence": 0.0,
            "Category Distribution": {}
        }

    df = pd.DataFrame(results)

    total_reviews = len(df)
    positive_reviews = int((df["Sentiment"] == "Positive").sum())
    negative_reviews = int((df["Sentiment"] == "Negative").sum())

    positive_pct = (positive_reviews / total_reviews * 100.0) if total_reviews > 0 else 0.0
    negative_pct = (negative_reviews / total_reviews * 100.0) if total_reviews > 0 else 0.0

    high_priority = int((df["Priority"] == "High").sum()) if "Priority" in df.columns else 0
    medium_priority = int((df["Priority"] == "Medium").sum()) if "Priority" in df.columns else 0
    low_priority = int((df["Priority"] == "Low").sum()) if "Priority" in df.columns else 0

    avg_confidence = float(df["Confidence"].mean()) if "Confidence" in df.columns else 0.0

    category_dist = {}
    if "Category" in df.columns:
        category_dist = df["Category"].value_counts().to_dict()

    summary = {
        "Total Reviews": total_reviews,
        "Positive Reviews": positive_reviews,
        "Negative Reviews": negative_reviews,
        "Positive Percentage": round(positive_pct, 2),
        "Negative Percentage": round(negative_pct, 2),
        "High Priority Reviews": high_priority,
        "Medium Priority Reviews": medium_priority,
        "Low Priority Reviews": low_priority,
        "Average Confidence": round(avg_confidence, 2),
        "Category Distribution": category_dist
    }

    return summary


# ============================================================
# SELF-TEST / CLI DEMO
# ============================================================

if __name__ == "__main__":
    test_cases = [
        {
            "review": "The headset stopped working after 3 days and the sound is cracked and horrible!",
            "sentiment": "Negative",
            "confidence": 98.4
        },
        {
            "review": "My order never arrived. Tracking says delivered 2 weeks ago but nothing in mailbox.",
            "sentiment": "Negative",
            "confidence": 94.2
        },
        {
            "review": "Customer service was completely rude and refused my return request.",
            "sentiment": "Negative",
            "confidence": 91.0
        },
        {
            "review": "This camera is absolutely stunning! Crisp photos and battery lasts forever.",
            "sentiment": "Positive",
            "confidence": 99.1
        },
        {
            "review": "It is okay, might be a bit pricey for the plastic finish.",
            "sentiment": "Negative",
            "confidence": 62.0
        }
    ]

    print("=" * 60)
    print("CUSTOMER REVIEW BUSINESS ANALYSIS TEST")
    print("=" * 60)

    analyzed = []
    for item in test_cases:
        enriched = analyze_prediction(item)
        analyzed.append(enriched)
        print(f"\nReview: \"{enriched['Review']}\"")
        print(f"  -> Sentiment   : {enriched['Sentiment']} ({enriched['Confidence']}%)")
        print(f"  -> Priority    : {enriched['Priority']} | SLA: {enriched['SLA_Target']}")
        print(f"  -> Category    : {enriched['Category']}")
        print(f"  -> Action      : {enriched['Recommended_Action']}")

    print("\n" + "=" * 60)
    print("EXECUTIVE BUSINESS SUMMARY")
    print("=" * 60)
    summary = create_business_summary(analyzed)
    for k, v in summary.items():
        print(f"  {k}: {v}")