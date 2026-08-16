"""
Noise x Balancing Experiment — with CV metric and RUS-excluded comparison
============================================================================
Same experiment grid as before. Adds three convergence score variants so
they can be compared side by side:

  1. ORIGINAL   — all 6 balancing methods, raw std (what we ran yesterday)
  2. RUS-EXCLUDED — 5 methods (RUS dropped), raw std
  3. RUS-EXCLUDED + CV — 5 methods, coefficient of variation (std/mean)
     instead of raw std, to correct for the floor-effect problem

CSV filename kept as "data.csv" as requested — point it at whichever
dataset file you rename to that.
"""

import string
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

from noise_injection import inject_random_noise, verify_noise_injection

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ------------------------------------------------------------
# DATASET-SPECIFIC LOADER
# ------------------------------------------------------------

def load_and_preprocess_amazon(path="data.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"reviews.text": "reviews", "reviews.username": "username"})

    to_drop = ['dateAdded', 'dateUpdated', 'name', 'asins', 'brand', 'categories',
               'primaryCategories', 'imageURLs', 'keys', 'manufacturer',
               'manufacturerNumber', 'reviews.date', 'reviews.dateAdded',
               'reviews.dateSeen', 'reviews.doRecommend', 'reviews.id',
               'reviews.numHelpful', 'reviews.rating', 'reviews.sourceURLs',
               'reviews.title', 'sourceURLs']
    df.drop(to_drop, inplace=True, axis=1, errors='ignore')

    rows_before = len(df)
    df = df.drop_duplicates(subset=['reviews'])
    print(f"Deduplication: {rows_before} -> {len(df)} rows "
          f"({rows_before - len(df)} duplicates removed)")

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


# ------------------------------------------------------------
# REUSABLE — dataset-agnostic from here
# ------------------------------------------------------------

BALANCERS = {
    "None": None,
    "ROS": RandomOverSampler(random_state=0),
    "RUS": RandomUnderSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
}


def run_noise_balancing_experiment(
    X_text, y_clean, target_class,
    noise_rates=(0.0, 0.10, 0.25, 0.40),
    random_state=100
):
    other_classes = [c for c in y_clean.unique() if c != target_class]

    X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
        X_text, y_clean, test_size=0.20, random_state=random_state, stratify=y_clean
    )

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()

    all_results = []

    for noise_rate in noise_rates:
        print(f"\n{'='*60}\nNOISE RATE: {noise_rate:.0%}\n{'='*60}")

        if noise_rate == 0.0:
            y_train_run = y_train_clean.copy()
        else:
            y_train_run, _ = inject_random_noise(
                y_train_clean, noise_rate, target_class, other_classes,
                random_state=random_state
            )
            verify_noise_injection(y_train_clean, y_train_run, target_class, noise_rate)

        for balance_name, balancer in BALANCERS.items():
            if balancer is None:
                X_tr, y_tr = X_train, y_train_run
            else:
                try:
                    X_tr, y_tr = balancer.fit_resample(X_train, y_train_run)
                except ValueError as e:
                    print(f"  Skipping {balance_name} at noise={noise_rate}: {e}")
                    continue

            for model_name, model in [
                ("SVM", LinearSVC(max_iter=2000)),
                ("NN", MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300,
                                      activation='relu', solver='adam', random_state=1)),
            ]:
                print(f"  Fitting {model_name} + {balance_name} "
                      f"(train size: {len(y_tr)})...", flush=True)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_test)
                print(f"    done.", flush=True)

                report = classification_report(
                    y_test_clean, y_pred,
                    labels=list(y_clean.unique()),
                    output_dict=True, zero_division=0
                )

                all_results.append({
                    "Noise Rate": noise_rate,
                    "Model": model_name,
                    "Balancing": balance_name,
                    "Accuracy": accuracy_score(y_test_clean, y_pred),
                    "Macro F1": f1_score(y_test_clean, y_pred, average="macro"),
                    "Weighted F1": f1_score(y_test_clean, y_pred, average="weighted"),
                    "Target Recall": report[target_class]["recall"],
                    "Target Precision": report[target_class]["precision"],
                })

    return pd.DataFrame(all_results)


# ------------------------------------------------------------
# THREE CONVERGENCE SCORE VARIANTS
# ------------------------------------------------------------

def compute_convergence_original(results_df):
    """Variant 1: all 6 methods, raw standard deviation (yesterday's version)."""
    out = (
        results_df.groupby(["Noise Rate", "Model"])["Target Recall"]
        .agg(mean_recall="mean", std_recall="std")
        .reset_index()
    )
    out["Variant"] = "1_original_all6_std"
    return out


def compute_convergence_rus_excluded(results_df):
    """Variant 2: drop RUS (mechanically different), raw standard deviation."""
    filtered = results_df[results_df["Balancing"] != "RUS"]
    out = (
        filtered.groupby(["Noise Rate", "Model"])["Target Recall"]
        .agg(mean_recall="mean", std_recall="std")
        .reset_index()
    )
    out["Variant"] = "2_rus_excluded_std"
    return out


def compute_convergence_cv(results_df):
    """
    Variant 3: drop RUS, use Coefficient of Variation (std/mean) instead
    of raw std. CV corrects for the floor effect — a low std only counts
    as "real agreement" if the shared value isn't itself near zero.

    NOTE: if mean_recall is exactly 0 (every method gave 0 recall), CV is
    mathematically undefined (0/0). We flag this explicitly as
    'all_zero_floor' rather than silently producing NaN or a fake 0.
    """
    filtered = results_df[results_df["Balancing"] != "RUS"]
    grouped = (
        filtered.groupby(["Noise Rate", "Model"])["Target Recall"]
        .agg(mean_recall="mean", std_recall="std")
        .reset_index()
    )

    grouped["all_zero_floor"] = grouped["mean_recall"] == 0
    grouped["CV"] = np.where(
        grouped["all_zero_floor"],
        np.nan,
        grouped["std_recall"] / grouped["mean_recall"]
    )
    grouped["Variant"] = "3_rus_excluded_CV"
    return grouped


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    print("Step 1: Loading and preprocessing dataset (data.csv)...")
    df_raw = load_and_preprocess_amazon("data.csv")

    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)
    print("Clean class distribution:\n", df_labeled['sentiment'].value_counts())

    print("\nStep 3: Running noise x balancing experiment grid...")
    results = run_noise_balancing_experiment(
        X_text=df_labeled['reviews_text'],
        y_clean=df_labeled['sentiment'],
        target_class="negative",
        noise_rates=(0.0, 0.10, 0.25, 0.40),
    )
    results.to_csv("noise_balancing_results.csv", index=False)
    print("\nSaved full per-method results to noise_balancing_results.csv")

    print("\nStep 4: Computing all three convergence score variants...")
    v1 = compute_convergence_original(results)
    v2 = compute_convergence_rus_excluded(results)
    v3 = compute_convergence_cv(results)

    print("\n--- Variant 1: Original (6 methods, std) ---")
    print(v1.to_string(index=False))

    print("\n--- Variant 2: RUS excluded (5 methods, std) ---")
    print(v2.to_string(index=False))

    print("\n--- Variant 3: RUS excluded + CV ---")
    print(v3.to_string(index=False))

    # Combine into one file for easy side-by-side comparison
    combined = pd.concat([v1, v2, v3], ignore_index=True)
    combined.to_csv("convergence_scores_all_variants.csv", index=False)
    print("\nSaved all three variants to convergence_scores_all_variants.csv")

    # Quick check: does std/CV increase monotonically with noise, per variant?
    print("\n--- Quick check: does the metric increase with noise? ---")
    for variant_name, df_v, metric_col in [
        ("Variant 1 (orig std)", v1, "std_recall"),
        ("Variant 2 (RUS-excl std)", v2, "std_recall"),
        ("Variant 3 (CV)", v3, "CV"),
    ]:
        for model in df_v["Model"].unique():
            sub = df_v[df_v["Model"] == model].sort_values("Noise Rate")
            values = sub[metric_col].tolist()
            increasing = all(a <= b for a, b in zip(values, values[1:]))
            print(f"{variant_name} | {model}: {values} | "
                  f"monotonically increasing: {increasing}")


if __name__ == "__main__":
    main()