"""
H2 Multi-Seed Validation (6 seeds)
====================================
Same experiment as H2 (systematic noise, negative class), repeated
across 6 independent random seeds instead of a single run. Reuses
seeds 1-6 from H1's multiseed run, so if you ever want to compare
random vs. systematic noise on the EXACT same train/test splits, you
can — same seed means same split, isolating noise TYPE as the only
variable, same discipline as everything else in this project.

Workflow, identical structure to H1 multiseed:
  1. For each seed: re-split the data, re-inject SYSTEMATIC noise,
     re-run the full balancing grid, compute CV.
  2. Aggregate across seeds: mean + 95% confidence interval.

Requires in the same folder: data.csv, systematic_noise_injection.py
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
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

from systematic_noise_injection import inject_systematic_noise, verify_systematic_injection

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ------------------------------------------------------------
# LOADER — identical across all experiments
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
# SINGLE-SEED RUN — H2's logic, called once per seed
# ------------------------------------------------------------

def run_single_seed(X_text, y_clean, target_class, seed, noise_rates=(0.0, 0.10, 0.25, 0.40)):
    other_classes = [c for c in y_clean.unique() if c != target_class]

    X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
        X_text, y_clean, test_size=0.20, random_state=seed, stratify=y_clean
    )

    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    seed_results = []

    for noise_rate in noise_rates:
        if noise_rate == 0.0:
            y_train_run = y_train_clean.copy()
        else:
            # Systematic noise needs X_train_text (not X_test, not full X_text) —
            # same split, same index alignment as y_train_clean
            y_train_run, _ = inject_systematic_noise(
                y_train_clean, X_train_text, noise_rate, target_class, other_classes,
                random_state=seed
            )
            verify_systematic_injection(
                y_train_clean, y_train_run, X_train_text, target_class, noise_rate
            )

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
                    "Seed": seed, "Noise Rate": noise_rate,
                    "Model": model_name, "Balancing": balance_name,
                    "Target Recall": report[target_class]["recall"],
                })

    return pd.DataFrame(seed_results)


# ------------------------------------------------------------
# AGGREGATE ACROSS SEEDS — identical logic to H1 multiseed, verified
# ------------------------------------------------------------

def compute_cv_per_seed(all_seeds_df):
    filtered = all_seeds_df[all_seeds_df["Balancing"] != "RUS"]
    grouped = (
        filtered.groupby(["Seed", "Noise Rate", "Model"])["Target Recall"]
        .agg(mean_recall="mean", std_recall="std")
        .reset_index()
    )
    grouped["CV"] = np.where(
        grouped["mean_recall"] == 0, np.nan,
        grouped["std_recall"] / grouped["mean_recall"]
    )
    return grouped


def compute_confidence_intervals(cv_per_seed_df):
    results = []
    for (noise_rate, model), group in cv_per_seed_df.groupby(["Noise Rate", "Model"]):
        cv_values = group["CV"].dropna().values
        n = len(cv_values)
        if n < 2:
            continue
        mean_cv = np.mean(cv_values)
        std_cv = np.std(cv_values, ddof=1)
        sem = std_cv / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * sem
        results.append({
            "Noise Rate": noise_rate, "Model": model,
            "Mean CV": mean_cv, "CI Lower": mean_cv - margin, "CI Upper": mean_cv + margin,
            "N Seeds": n,
        })
    return pd.DataFrame(results)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    SEEDS = [1, 2, 3, 4, 5, 6]  # reusing seeds 1-6 from H1's multiseed run

    print("Step 1: Loading and preprocessing dataset...")
    df_raw = load_and_preprocess_amazon("data.csv")

    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)

    print(f"\nStep 3: Running H2 (systematic noise) across {len(SEEDS)} seeds: {SEEDS}")
    all_runs = []
    for seed in SEEDS:
        start = time.time()
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        seed_df = run_single_seed(
            X_text=df_labeled['reviews_text'], y_clean=df_labeled['sentiment'],
            target_class="negative", seed=seed
        )
        all_runs.append(seed_df)
        print(f"  Seed {seed} completed in {time.time()-start:.1f}s")

    full_results = pd.concat(all_runs, ignore_index=True)
    full_results.to_csv("h2_multiseed_full_results.csv", index=False)
    print("\nSaved full per-seed results to h2_multiseed_full_results.csv")

    print("\nStep 4: Computing CV per seed...")
    cv_per_seed = compute_cv_per_seed(full_results)
    cv_per_seed.to_csv("h2_multiseed_cv_per_seed.csv", index=False)

    print("\nStep 5: Computing mean + 95% confidence intervals across seeds...")
    ci_results = compute_confidence_intervals(cv_per_seed)
    ci_results.to_csv("h2_multiseed_confidence_intervals.csv", index=False)
    print(ci_results.to_string(index=False))

    print("\n--- Does the 0% CI overlap with the 40% CI? (the key question) ---")
    for model in ci_results["Model"].unique():
        sub = ci_results[ci_results["Model"] == model]
        low_noise = sub[sub["Noise Rate"] == 0.0]
        high_noise = sub[sub["Noise Rate"] == 0.40]
        if len(low_noise) and len(high_noise):
            no_overlap = low_noise["CI Upper"].values[0] < high_noise["CI Lower"].values[0]
            print(f"{model}: 0% CI upper = {low_noise['CI Upper'].values[0]:.3f}, "
                  f"40% CI lower = {high_noise['CI Lower'].values[0]:.3f} | "
                  f"Statistically distinct (no overlap): {no_overlap}")


if __name__ == "__main__":
    main()
