"""
H2: Does SYSTEMATIC noise (deterministic selection — shortest reviews
corrupted first) produce a different convergence signal than RANDOM
noise (H1), at the same corruption rates, same target class (negative)?

Combines:
  - data.csv (34,660-row dataset, current schema)
  - inject_systematic_noise (deterministic selection)
  - The H1 experiment framework (balancing grid, CV convergence score)

Fixes carried forward: LinearSVC (not SVC), sparse matrices (no .toarray()),
progress printing, correct to_drop list for this dataset's columns.
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

from systematic_noise_injection import inject_systematic_noise, verify_systematic_injection

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# ------------------------------------------------------------
# LOADER — correct schema for the 34,660-row dataset
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


BALANCERS = {
    "None": None,
    "ROS": RandomOverSampler(random_state=0),
    "RUS": RandomUnderSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
}


def run_systematic_noise_experiment(
    X_text, y_clean, target_class,
    noise_rates=(0.0, 0.10, 0.25, 0.40),
    random_state=100
):
    other_classes = [c for c in y_clean.unique() if c != target_class]

    X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
        X_text, y_clean, test_size=0.20, random_state=random_state, stratify=y_clean
    )

    # Sparse — no .toarray()
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    all_results = []

    for noise_rate in noise_rates:
        print(f"\n{'='*60}\nNOISE RATE: {noise_rate:.0%} (SYSTEMATIC)\n{'='*60}")

        if noise_rate == 0.0:
            y_train_run = y_train_clean.copy()
        else:
            # Systematic injection needs the TEXT to select shortest reviews —
            # this is the one structural difference from the H1/H3 scripts
            y_train_run, _ = inject_systematic_noise(
                y_train_clean, X_train_text, noise_rate, target_class, other_classes,
                random_state=random_state
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
                except ValueError as e:
                    print(f"  Skipping {balance_name} at noise={noise_rate}: {e}")
                    continue

            for model_name, model in [
                ("SVM", LinearSVC(max_iter=2000)),
                ("NN", MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300,
                                      activation='relu', solver='adam', random_state=1)),
            ]:
                print(f"  Fitting {model_name} + {balance_name} "
                      f"(train size: {y_tr.shape[0]})...", flush=True)
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


def compute_convergence_cv(results_df):
    filtered = results_df[results_df["Balancing"] != "RUS"]
    grouped = (
        filtered.groupby(["Noise Rate", "Model"])["Target Recall"]
        .agg(mean_recall="mean", std_recall="std")
        .reset_index()
    )
    grouped["all_zero_floor"] = grouped["mean_recall"] == 0
    grouped["CV"] = np.where(
        grouped["all_zero_floor"], np.nan,
        grouped["std_recall"] / grouped["mean_recall"]
    )
    return grouped


def main():
    print("Step 1: Loading and preprocessing dataset (data.csv)...")
    df_raw = load_and_preprocess_amazon("data.csv")

    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)
    print("Clean class distribution:\n", df_labeled['sentiment'].value_counts())

    print("\nStep 3: Running H2 experiment (SYSTEMATIC noise, target_class = negative)...")
    results = run_systematic_noise_experiment(
        X_text=df_labeled['reviews_text'],
        y_clean=df_labeled['sentiment'],
        target_class="negative",
        noise_rates=(0.0, 0.10, 0.25, 0.40),
    )
    results.to_csv("h2_systematic_noise_results.csv", index=False)
    print("\nSaved to h2_systematic_noise_results.csv")

    print("\nStep 4: Computing CV convergence score...")
    cv_results = compute_convergence_cv(results)
    cv_results.to_csv("h2_systematic_convergence_cv.csv", index=False)
    print(cv_results.to_string(index=False))

    print("\n--- Does CV increase monotonically with noise? ---")
    for model in cv_results["Model"].unique():
        sub = cv_results[cv_results["Model"] == model].sort_values("Noise Rate")
        values = sub["CV"].tolist()
        increasing = all(a <= b for a, b in zip(values, values[1:]))
        print(f"{model}: {values} | monotonically increasing: {increasing}")

    print("\n--- Comparison note ---")
    print("Compare h2_systematic_convergence_cv.csv against H1's")
    print("convergence_scores_all_variants.csv (Variant 3, target_class=negative)")
    print("to see whether systematic noise produces a SHARPER signal than random noise.")


if __name__ == "__main__":
    main()