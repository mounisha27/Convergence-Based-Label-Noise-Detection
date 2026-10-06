"""
H5: Does the CV Label-Noise Diagnostic Survive a Representation Shift?
=========================================================================
Tests whether the CV metric's noise-tracking behavior is representation-
dependent, using three conditions that isolate dimensionality from
semantic density:

  Condition A: TF-IDF, full dimensionality (2,500-dim)      -- existing baseline
  Condition B: TF-IDF, reduced via TruncatedSVD to 384-dim  -- isolates dim. reduction alone
  Condition C: MiniLM sentence embeddings, native 384-dim   -- isolates semantic density

Incorporates two critical fixes from external methodological review:
  1. CV is reported ALONGSIDE raw std and spread (max-min), not alone,
     since higher baseline recall (denominator) can shrink CV even when
     absolute disagreement is unchanged -- verified as a real artifact
     before writing this script.
  2. Per-resampler recall is tracked individually, not just the aggregate
     CV, to identify which specific method drives any observed change.

Requires: data.csv, noise_injection.py, sentence-transformers
"""

import string
import time
import pandas as pd
import numpy as np
from scipy import stats
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

from sentence_transformers import SentenceTransformer

from noise_injection import inject_random_noise, verify_noise_injection

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ------------------------------------------------------------
# LOADER -- same as all prior experiments
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

    df['reviews_clean'] = df['reviews'].astype(str).apply(remove_punctuations)
    df['reviews_tokens'] = df['reviews_clean'].apply(word_tokenize)
    stop = set(stopwords.words('english'))
    df['reviews_tokens'] = df['reviews_tokens'].apply(lambda x: [w for w in x if w.lower() not in stop])
    return df


def senti_pol_fixed(tokens):
    return TextBlob(" ".join(tokens)).sentiment.polarity


def assign_labels(df, senti_pol_fn):
    df = df.copy()
    df['senti_polarity'] = df['reviews_tokens'].apply(senti_pol_fn)
    condition = [
        df['senti_polarity'] > 0.05,
        (df['senti_polarity'] <= 0.05) & (df['senti_polarity'] > -0.05),
        df['senti_polarity'] <= -0.05
    ]
    df['sentiment'] = np.select(condition, ['positive', 'neutral', 'negative'], default='neutral')
    # Two text representations kept: cleaned/tokenized (for TF-IDF) and
    # raw original text (for MiniLM, which expects natural sentences)
    df['reviews_text_clean'] = [" ".join(tokens) for tokens in df['reviews_tokens']]
    return df


BALANCERS = {
    "None": None, "ROS": RandomOverSampler(random_state=0),
    "SMOTE": SMOTE(random_state=42), "ADASYN": ADASYN(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
}


# ------------------------------------------------------------
# FEATURE EXTRACTION -- three conditions, isolating dimensionality from semantics
# ------------------------------------------------------------

def get_features_condition_A(train_text, test_text):
    """TF-IDF, full 2,500 dimensions -- existing baseline representation."""
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)
    return X_train, X_test


def get_features_condition_B(train_text, test_text, seed):
    """TF-IDF reduced via TruncatedSVD to 384-dim -- isolates dimensionality
    reduction alone, without introducing semantic/contextual information."""
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train_full = vectorizer.fit_transform(train_text)
    X_test_full = vectorizer.transform(test_text)
    svd = TruncatedSVD(n_components=384, random_state=seed)
    X_train = svd.fit_transform(X_train_full)
    X_test = svd.transform(X_test_full)
    return X_train, X_test


_minilm_model = None

def get_features_condition_C(train_text, test_text):
    """MiniLM sentence embeddings, native 384-dim -- isolates semantic
    density. Model is loaded once and reused across seeds for efficiency."""
    global _minilm_model
    if _minilm_model is None:
        print("  Loading MiniLM model (first time only)...")
        _minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
    X_train = _minilm_model.encode(list(train_text), show_progress_bar=False)
    X_test = _minilm_model.encode(list(test_text), show_progress_bar=False)
    return X_train, X_test


FEATURE_CONDITIONS = {
    "A_TFIDF_full": lambda tr, te, seed: get_features_condition_A(tr, te),
    "B_TFIDF_SVD384": lambda tr, te, seed: get_features_condition_B(tr, te, seed),
    "C_MiniLM_384": lambda tr, te, seed: get_features_condition_C(tr, te),
}


# ------------------------------------------------------------
# SINGLE-SEED RUN, per feature condition
# ------------------------------------------------------------

def run_single_seed(df_labeled, target_class, seed, condition_name, condition_fn,
                     noise_rates=(0.0, 0.10, 0.25, 0.40)):
    y_clean = df_labeled['sentiment']
    other_classes = [c for c in y_clean.unique() if c != target_class]

    # MiniLM works on natural text; TF-IDF conditions use cleaned/tokenized text.
    # Using cleaned text consistently across all three conditions keeps the
    # noise-injection and split logic identical; only the FEATURE EXTRACTION differs.
    text_col = df_labeled['reviews_text_clean']

    X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
        text_col, y_clean, test_size=0.20, random_state=seed, stratify=y_clean
    )

    X_train, X_test = condition_fn(X_train_text, X_test_text, seed)

    seed_results = []
    for noise_rate in noise_rates:
        if noise_rate == 0.0:
            y_train_run = y_train_clean.copy()
        else:
            y_train_run, _ = inject_random_noise(
                y_train_clean, noise_rate, target_class, other_classes, random_state=seed
            )
            verify_noise_injection(y_train_clean, y_train_run, target_class, noise_rate)

        for balance_name, balancer in BALANCERS.items():
            if balancer is None:
                X_tr, y_tr = X_train, y_train_run
            else:
                try:
                    X_tr, y_tr = balancer.fit_resample(X_train, y_train_run)
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
                    y_test_clean, y_pred, labels=list(y_clean.unique()),
                    output_dict=True, zero_division=0
                )
                seed_results.append({
                    "Condition": condition_name, "Seed": seed, "Noise Rate": noise_rate,
                    "Model": model_name, "Balancing": balance_name,
                    "Target Recall": report[target_class]["recall"],
                })

    return pd.DataFrame(seed_results)


# ------------------------------------------------------------
# AGGREGATION -- CV decomposed into mean, raw std, spread (critical fix)
# ------------------------------------------------------------

def compute_cv_decomposed_per_seed(all_seeds_df):
    filtered = all_seeds_df  # RUS not used in this study; no exclusion needed
    grouped = (
        filtered.groupby(["Condition", "Seed", "Noise Rate", "Model"])["Target Recall"]
        .agg(mean_recall="mean", std_recall="std",
             min_recall="min", max_recall="max")
        .reset_index()
    )
    grouped["spread"] = grouped["max_recall"] - grouped["min_recall"]
    grouped["CV"] = np.where(
        grouped["mean_recall"] == 0, np.nan,
        grouped["std_recall"] / grouped["mean_recall"]
    )
    return grouped


def compute_confidence_intervals(decomposed_df, value_col="CV"):
    results = []
    for (condition, noise_rate, model), group in decomposed_df.groupby(["Condition", "Noise Rate", "Model"]):
        values = group[value_col].dropna().values
        n = len(values)
        if n < 2:
            continue
        mean_v = np.mean(values)
        std_v = np.std(values, ddof=1)
        sem = std_v / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * sem
        results.append({
            "Condition": condition, "Noise Rate": noise_rate, "Model": model,
            f"Mean {value_col}": mean_v, "CI Lower": mean_v - margin, "CI Upper": mean_v + margin,
            "N Seeds": n,
        })
    return pd.DataFrame(results)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    SEEDS = [1, 2, 3, 4, 5]  # 5 seeds per condition; 3 conditions = 15 full runs total
    TARGET_CLASS = "negative"

    print("Step 1: Loading and preprocessing dataset...")
    df_raw = load_and_preprocess_amazon("data.csv")
    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)

    all_results = []
    for condition_name, condition_fn in FEATURE_CONDITIONS.items():
        print(f"\n{'#'*60}\n# CONDITION: {condition_name}\n{'#'*60}")
        for seed in SEEDS:
            start = time.time()
            print(f"\n--- {condition_name} - Seed {seed} ---")
            seed_df = run_single_seed(
                df_labeled, TARGET_CLASS, seed, condition_name, condition_fn
            )
            all_results.append(seed_df)
            print(f"  Completed in {time.time()-start:.1f}s")

    full_results = pd.concat(all_results, ignore_index=True)
    full_results.to_csv("h5_representation_shift_full_results.csv", index=False)
    print("\nSaved full results to h5_representation_shift_full_results.csv")

    print("\nStep: Computing decomposed CV (mean, std, spread) per seed...")
    decomposed = compute_cv_decomposed_per_seed(full_results)
    decomposed.to_csv("h5_representation_shift_decomposed.csv", index=False)

    print("\nStep: Computing confidence intervals for CV, std, and spread separately...")
    for metric in ["CV", "std_recall", "spread"]:
        ci_df = compute_confidence_intervals(decomposed, value_col=metric)
        ci_df.to_csv(f"h5_ci_{metric}.csv", index=False)
        print(f"\n--- {metric} across conditions ---")
        print(ci_df.to_string(index=False))

    print("\n--- Per-resampler recall breakdown (mechanistic diagnostic) ---")
    per_resampler = (
        full_results.groupby(["Condition", "Noise Rate", "Model", "Balancing"])["Target Recall"]
        .mean().reset_index()
    )
    per_resampler.to_csv("h5_per_resampler_breakdown.csv", index=False)
    print(per_resampler.to_string(index=False))


if __name__ == "__main__":
    main()
