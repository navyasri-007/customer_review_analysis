import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# PROJECT & MODULE PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# ROBUST DATASET PATH RESOLVER
# ============================================================

def get_dataset_file(filename):
    """
    Locates dataset text file, handling both flat file structures
    and nested directory extractions (e.g. data/amazonreviews/train.ft.txt/train.ft.txt).
    """
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "amazonreviews", filename, filename),
        os.path.join(PROJECT_ROOT, "data", "amazonreviews", filename),
        os.path.join(PROJECT_ROOT, "data", filename, filename),
        os.path.join(PROJECT_ROOT, "data", filename),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback to direct path for error messaging
    return os.path.join(PROJECT_ROOT, "data", "amazonreviews", filename)


TRAIN_PATH = get_dataset_file("train.ft.txt")
TEST_PATH = get_dataset_file("test.ft.txt")


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset_files():
    """Verifies that dataset files exist and are readable."""
    print("=" * 60)
    print("CHECKING DATASET FILES")
    print("=" * 60)

    train_resolved = get_dataset_file("train.ft.txt")
    test_resolved = get_dataset_file("test.ft.txt")

    print("\nResolved Training file:")
    print(f"  {train_resolved}")
    print(f"  Exists and is file: {os.path.isfile(train_resolved)}")

    print("\nResolved Testing file:")
    print(f"  {test_resolved}")
    print(f"  Exists and is file: {os.path.isfile(test_resolved)}")

    if not os.path.isfile(train_resolved):
        raise FileNotFoundError(
            f"Training file not found at any candidate location:\n{train_resolved}"
        )

    if not os.path.isfile(test_resolved):
        raise FileNotFoundError(
            f"Testing file not found at any candidate location:\n{test_resolved}"
        )

    return train_resolved, test_resolved


# ============================================================
# READ SAMPLE
# ============================================================

def read_sample(n=5):
    """Reads and prints the first n sample raw lines from the training file."""
    train_file = get_dataset_file("train.ft.txt")
    print(f"\nReading first {n} sample reviews from {os.path.basename(train_file)}...")

    sample_lines = []
    with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
        for i in range(n):
            line = f.readline().strip()
            if line:
                sample_lines.append(line)
                print(f"[{i + 1}] {line[:120]}...")

    return sample_lines


# ============================================================
# LOAD BALANCED DATASET
# ============================================================

def load_reviews(max_per_class=10000, source_file=None):
    """
    Loads a balanced dataset of positive and negative reviews.
    
    __label__1 -> 0 (Negative)
    __label__2 -> 1 (Positive)
    """
    if source_file is None:
        source_file = get_dataset_file("train.ft.txt")

    positive_reviews = []
    negative_reviews = []

    print(f"\nLoading up to {max_per_class:,} reviews per class from:\n  {source_file}")

    with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                label, review = line.split(" ", 1)
            except ValueError:
                continue

            # __label__1 = Negative (0)
            if label == "__label__1" and len(negative_reviews) < max_per_class:
                negative_reviews.append((0, review))

            # __label__2 = Positive (1)
            elif label == "__label__2" and len(positive_reviews) < max_per_class:
                positive_reviews.append((1, review))

            # Stop when both classes are satisfied
            if len(negative_reviews) >= max_per_class and len(positive_reviews) >= max_per_class:
                break

    print(f"  -> Negative reviews collected: {len(negative_reviews):,}")
    print(f"  -> Positive reviews collected: {len(positive_reviews):,}")

    # Combine and shuffle
    df = pd.DataFrame(
        negative_reviews + positive_reviews,
        columns=["label", "review"]
    )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ============================================================
# SPLIT DATASET
# ============================================================

def split_dataset(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Splits DataFrame into train, validation, and test subsets.
    Defaults to 70% Train, 15% Validation, 15% Test.
    """
    temp_ratio = val_ratio + test_ratio

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=df["label"]
    )

    val_relative_ratio = val_ratio / temp_ratio

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative_ratio,
        random_state=random_state,
        stratify=temp_df["label"]
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# COMPLETE DATA PREPARATION PIPELINE
# ============================================================

def prepare_dataset(max_per_class=10000):
    """
    Orchestrates dataset checking, balanced loading, and splitting.
    Returns (train_df, val_df, test_df).
    """
    check_dataset_files()
    read_sample(3)

    df = load_reviews(max_per_class=max_per_class)

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)
    print(f"Total reviews loaded: {len(df):,}")
    print("\nClass distribution:")
    for label, count in df["label"].value_counts().items():
        name = "Positive (1)" if label == 1 else "Negative (0)"
        print(f"  {name}: {count:,} ({count / len(df) * 100:.1f}%)")

    train_df, val_df, test_df = split_dataset(df)

    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)
    print(f"  Training set   : {len(train_df):,} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set : {len(val_df):,} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set       : {len(test_df):,} examples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Customer Review Dataset Preprocessing")
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=1000,
        help="Number of samples per class for verification (default: 1000)"
    )
    args = parser.parse_args()

    print("Running Customer Review Data Preprocessor...")
    train_df, val_df, test_df = prepare_dataset(max_per_class=args.max_per_class)
    print("\nData preprocessing completed successfully.")