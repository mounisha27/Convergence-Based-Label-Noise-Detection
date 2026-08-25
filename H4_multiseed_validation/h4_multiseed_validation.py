"""
H4 Multi-Seed Validation (5 seeds)
====================================
Same comparison as H4 (Confident Learning vs. CV, negative class),
repeated across 5 independent seeds instead of a single run. Same
structure as H1/H2 multiseed: for each seed, re-split, re-inject
noise, re-run both cleanlab and the CV grid, then aggregate.

Requires in the same folder: data.csv, noise_injection.py, cleanlab
"""

import time
import string
import pandas as pd
import numpy as np
from scipy import stats
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
    return TextBlob(" ".join(tokens)).sentiment.polarity


def assign_labels(df, senti_pol_fn):
    df = df.copy()
    df['senti_polarity'] = df['reviews'].apply(senti_pol_fn)
    condition = [
        df['senti_polarity'] > 0.05,
        (df['senti_polarity'] <= 0.05) & (df['senti_polarity'] > -0.05),
        df['senti_polarity'] <= -0.05
    ]
    df['sentiment'] = np.select(condition, ['positive', 'neutral', 'negative'], default='neutral')
    df['reviews_text'] = [" ".join(review) for review in df['reviews']]
    return df


BALANCERS = {
    "None": None, "ROS": RandomOverSampler(random_state=0),
    "SMOTE": SMOTE(random_state=42), "ADASYN": ADASYN(random_state=42),
    "Borderline-SMOTE": BorderlineSMOTE(random_state=42),
}


def run_cleanlab_with_ground_truth(X_train_text, y_train_noisy, corrupt_indices):
    vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
    X_features = vectorizer.fit_transform(X_train_text)
    model = LogisticRegression(max_iter=1000)
    pred_probs = cross_val_predict(model, X_features, y_train_noisy, cv=5, method="predict_proba")

    sorted_classes = sorted(y_train_noisy.unique())
    label_map = {label: i for i, label in enumerate(sorted_classes)}
    y_int = y_train_noisy.map(label_map).values

    flagged_positions = cleanlab.filter.find_label_issues(
        labels=y_int, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    flagged_index_labels = set(y_train_noisy.index[flagged_positions])
    true_corrupted = set(corrupt_indices)
    true_positives = flagged_index_labels & true_corrupted
    precision = len(true_positives) / len(flagged_index_labels) if flagged_index_labels else 0.0
    recall = len(true_positives) / len(true_corrupted) if true_corrupted else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_cv_grid(X_train, y_train_noisy, X_test, y_test_clean, target_class, class_labels):
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
            report = classification_report(y_test_clean, y_pred, labels=class_labels,
                                            output_dict=True, zero_division=0)
            results.append({"Target Recall": report[target_class]["recall"]})
    results_df = pd.DataFrame(results)
    mean_recall = results_df["Target Recall"].mean()
    std_recall = results_df["Target Recall"].std()
    cv = std_recall / mean_recall if mean_recall > 0 else np.nan
    return cv


def compute_confidence_intervals(values_dict):
    """values_dict: {noise_rate: [list of values across seeds]}"""
    results = []
    for noise_rate, values in values_dict.items():
        values = [v for v in values if not np.isnan(v)]
        n = len(values)
        if n < 2:
            continue
        mean_v = np.mean(values)
        std_v = np.std(values, ddof=1)
        sem = std_v / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * sem
        results.append({"Noise Rate": noise_rate, "Mean": mean_v,
                         "CI Lower": mean_v - margin, "CI Upper": mean_v + margin, "N Seeds": n})
    return pd.DataFrame(results)


def main():
    SEEDS = [1, 2, 3, 4, 5]
    noise_rates = (0.10, 0.25, 0.40)

    print("Step 1: Loading and preprocessing dataset...")
    df_raw = load_and_preprocess_amazon("data.csv")
    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)
    y_clean = df_labeled['sentiment']
    class_labels = list(y_clean.unique())
    target_class = "negative"
    other_classes = [c for c in class_labels if c != target_class]

    all_results = []

    for seed in SEEDS:
        seed_start = time.time()
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

        X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
            df_labeled['reviews_text'], y_clean, test_size=0.20, random_state=seed, stratify=y_clean
        )
        vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8)
        X_train_vec = vectorizer.fit_transform(X_train_text)
        X_test_vec = vectorizer.transform(X_test_text)

        for noise_rate in noise_rates:
            y_train_noisy, corrupt_indices = inject_random_noise(
                y_train_clean, noise_rate, target_class, other_classes, random_state=seed
            )
            verify_noise_injection(y_train_clean, y_train_noisy, target_class, noise_rate)

            print(f"  [Seed {seed}, {noise_rate:.0%}] Running cleanlab...")
            cl_result = run_cleanlab_with_ground_truth(X_train_text, y_train_noisy, corrupt_indices)

            print(f"  [Seed {seed}, {noise_rate:.0%}] Running CV grid...")
            cv_result = run_cv_grid(X_train_vec, y_train_noisy, X_test_vec, y_test_clean,
                                     target_class, class_labels)

            all_results.append({
                "Seed": seed, "Noise Rate": noise_rate,
                "CL Precision": cl_result["precision"], "CL Recall": cl_result["recall"],
                "CL F1": cl_result["f1"], "CV Score": cv_result,
            })

        print(f"  Seed {seed} completed in {time.time()-seed_start:.1f}s")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("h4_multiseed_full_results.csv", index=False)
    print("\nSaved full results to h4_multiseed_full_results.csv")

    print("\nComputing confidence intervals...")
    cv_by_rate = {r: results_df[results_df["Noise Rate"] == r]["CV Score"].tolist() for r in noise_rates}
    precision_by_rate = {r: results_df[results_df["Noise Rate"] == r]["CL Precision"].tolist() for r in noise_rates}

    cv_ci = compute_confidence_intervals(cv_by_rate)
    precision_ci = compute_confidence_intervals(precision_by_rate)

    cv_ci.to_csv("h4_multiseed_cv_ci.csv", index=False)
    precision_ci.to_csv("h4_multiseed_precision_ci.csv", index=False)

    print("\n--- CV Score, mean + 95% CI across seeds ---")
    print(cv_ci.to_string(index=False))
    print("\n--- Cleanlab Precision, mean + 95% CI across seeds ---")
    print(precision_ci.to_string(index=False))


if __name__ == "__main__":
    main()
