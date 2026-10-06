"""
Cleanlab vs. Convergence-CV: A Controlled Benchmark Comparison
==================================================================
Unlike the CV validation (H1/H2/H3), which only tested the DIRECTION
of a trend, this experiment has TRUE ground truth: since we control
the noise injection ourselves, we know exactly which examples were
corrupted. This lets us directly measure cleanlab's detection
accuracy (precision/recall against real corrupted examples) — not
just observe a trend, but score it against a known answer.

At each noise rate, we compute:
  1. Cleanlab's precision/recall at flagging the ACTUAL corrupted examples
  2. The CV convergence score (same metric as H1/H2/H3)
  3. Wall-clock time for each approach (a real cost comparison)

Research question: does the cheap, aggregate CV signal track the same
underlying noise that the more expensive, per-example cleanlab
detector directly measures?
"""

import time
import string
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

import cleanlab

from noise_injection import inject_random_noise, verify_noise_injection

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ------------------------------------------------------------
# LOADER (same as prior experiments)
# ------------------------------------------------------------

def load_and_preprocess_amazon(path="data.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"reviews.text": "reviews", "reviews.username": "username"})

    to_drop = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer',
               'reviews.date', 'reviews.dateAdded', 'reviews.dateSeen',
               'reviews.didPurchase', 'reviews.doRecommend', 'reviews.id',
               'reviews.numHelpful', 'reviews.rating', 'reviews.sourceURLs',
               'reviews.title', 'reviews.userCity', 'reviews.userProvince']
    df.drop(to_drop, inplace=True, axis=1, errors='ignore')

    rows_before = len(df)
    df = df.drop_duplicates(subset=['reviews'])
    print(f"Deduplication: {rows_before} -> {len(df)} rows")

    def remove_punctuations(review):
        for punctuation in string.punctuation:
            review = review.replace(punctuation, '')
        return review

    df['reviews'] = df['reviews'].astype(str).apply(remove_punctuations)
    df['reviews'] = df['reviews'].apply(word_tokenize)

    stop = set(stopwords.words('english'))
    df['reviews'] = df['reviews'].apply(lambda x: [w for w in x if w.lower() not in stop])

    return df


def senti_pol_fixed(tokens):
    full_review = " ".join(tokens)
    return TextBlob(full_review).sentiment.polarity


def assign_labels(df, senti_pol_fn):
    df = df.copy()
    df['senti_polarity'] = df['reviews'].apply(senti_pol_fn)
    condition = [
        df['senti_polarity'] > 0.05,
        (df['senti_polarity'] <= 0.05) & (df['senti_polarity'] > -0.05),
        df['senti_polarity'] <= -0.05
    ]
    values = ['positive', 'neutral', 'negative']
    df['sentiment'] = np.select(condition, values, default='neutral')
    df['reviews_text'] = [" ".join(review) for review in df['reviews']]
    return df


BALANCERS = {
    "None": None,
    "ROS": RandomOverSampler(random_state=0),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
}


# ------------------------------------------------------------
# CLEANLAB SIDE — with ground-truth precision/recall
# ------------------------------------------------------------

def run_cleanlab_with_ground_truth(X_train_text, y_train_noisy, corrupt_indices):
    """
    Runs cleanlab on the noisy training labels, and scores its detection
    against the TRUE set of corrupted indices (known because we injected
    the noise ourselves).
    """
    start = time.time()

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_features = vectorizer.fit_transform(X_train_text)

    model = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(model, X_features, y_train_noisy, cv=5, method="predict_proba")

    sorted_classes = sorted(y_train_noisy.unique())
    label_map = {label: i for i, label in enumerate(sorted_classes)}
    y_int = y_train_noisy.map(label_map).values

    flagged_positions = cleanlab.filter.find_label_issues(
        labels=y_int, pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence"
    )

    elapsed = time.time() - start

    # Map flagged POSITIONS (0-based, array order) back to actual pandas index labels
    flagged_index_labels = set(y_train_noisy.index[flagged_positions])
    true_corrupted = set(corrupt_indices)

    true_positives = flagged_index_labels & true_corrupted
    precision = len(true_positives) / len(flagged_index_labels) if flagged_index_labels else 0.0
    recall = len(true_positives) / len(true_corrupted) if true_corrupted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "n_flagged": len(flagged_index_labels),
        "n_true_corrupted": len(true_corrupted),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "time_seconds": elapsed,
    }


# ------------------------------------------------------------
# CV SIDE — same logic as H1/H2/H3, with timing added
# ------------------------------------------------------------

def run_cv_grid(X_train, y_train_noisy, X_test, y_test_clean, target_class, class_labels):
    start = time.time()
    results = []

    for balance_name, balancer in BALANCERS.items():
        if balancer is None:
            X_tr, y_tr = X_train, y_train_noisy
        else:
            try:
                X_tr, y_tr = balancer.fit_resample(X_train, y_train_noisy)
            except ValueError:
                continue

        for model_name, model in [
            ("SVM", LinearSVC(max_iter=2000)),
            ("NN", MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300,
                                  activation='relu', solver='adam', random_state=1)),
        ]:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test_clean, y_pred, labels=class_labels,
                output_dict=True, zero_division=0
            )
            results.append({
                "Balancing": balance_name, "Model": model_name,
                "Target Recall": report[target_class]["recall"],
            })

    elapsed = time.time() - start
    results_df = pd.DataFrame(results)

    mean_recall = results_df["Target Recall"].mean()
    std_recall = results_df["Target Recall"].std()
    cv = std_recall / mean_recall if mean_recall > 0 else np.nan

    return {"CV": cv, "mean_recall": mean_recall, "time_seconds": elapsed}


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("Step 1: Loading and preprocessing dataset...")
    df_raw = load_and_preprocess_amazon("data.csv")

    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)
    y_clean = df_labeled['sentiment']
    class_labels = list(y_clean.unique())
    target_class = "negative"
    other_classes = [c for c in class_labels if c != target_class]

    print("\nStep 3: Splitting (once) — test set stays clean throughout...")
    X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
        df_labeled['reviews_text'], y_clean, test_size=0.20, random_state=100, stratify=y_clean
    )

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    noise_rates = (0.10, 0.25, 0.40)
    all_comparison_results = []

    for noise_rate in noise_rates:
        print(f"\n{'='*60}\nNOISE RATE: {noise_rate:.0%}\n{'='*60}")

        y_train_noisy, corrupt_indices = inject_random_noise(
            y_train_clean, noise_rate, target_class, other_classes, random_state=100
        )
        verify_noise_injection(y_train_clean, y_train_noisy, target_class, noise_rate)

        print("  Running cleanlab (with ground-truth scoring)...")
        cleanlab_result = run_cleanlab_with_ground_truth(X_train_text, y_train_noisy, corrupt_indices)
        print(f"    Cleanlab: precision={cleanlab_result['precision']:.3f}, "
              f"recall={cleanlab_result['recall']:.3f}, "
              f"f1={cleanlab_result['f1']:.3f}, "
              f"time={cleanlab_result['time_seconds']:.1f}s")

        print("  Running CV balancing grid...")
        cv_result = run_cv_grid(X_train_vec, y_train_noisy, X_test_vec, y_test_clean,
                                 target_class, class_labels)
        print(f"    CV: {cv_result['CV']:.3f}, time={cv_result['time_seconds']:.1f}s")

        all_comparison_results.append({
            "Noise Rate": noise_rate,
            "Cleanlab Precision": cleanlab_result["precision"],
            "Cleanlab Recall": cleanlab_result["recall"],
            "Cleanlab F1": cleanlab_result["f1"],
            "Cleanlab Time (s)": cleanlab_result["time_seconds"],
            "CV Score": cv_result["CV"],
            "CV Time (s)": cv_result["time_seconds"],
        })

    results_df = pd.DataFrame(all_comparison_results)
    results_df.to_csv("cleanlab_vs_cv_comparison.csv", index=False)

    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\nSaved to cleanlab_vs_cv_comparison.csv")


if __name__ == "__main__":
    main()
