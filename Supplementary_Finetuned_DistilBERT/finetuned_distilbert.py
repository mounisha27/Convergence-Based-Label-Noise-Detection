"""
Supplementary: Fine-Tuned Transformer Classifier Under Label Noise
==================================================================
Addresses the second half of BZon's "dated architecture" critique:
H5 (h5_representation_shift.py) modernized the FEATURES (MiniLM
embeddings) but kept LinearSVC/MLP as classifiers. This script instead
fine-tunes DistilBERT directly -- a genuinely modern classifier, not
just a modern feature extractor feeding an old classifier.

IMPORTANT METHODOLOGICAL NOTE (for the paper):
Your 5 resampling methods (ROS, SMOTE, ADASYN, Borderline-SMOTE) work
by blending feature VECTORS together. Fine-tuning operates on raw
TEXT, tokenized fresh each time -- there is no clean vector to blend.
The standard substitute for class imbalance during fine-tuning is
CLASS-WEIGHTED LOSS (penalizing errors on the rare class more heavily),
not resampling. This script therefore does NOT reproduce the
5-condition CV grid; it tests whether a single modern classifier's
recall on the target class still degrades predictably with injected
noise, and by how much a class-weighted loss can compensate.

Requires: data.csv, noise_injection.py, transformers, torch
Runtime: CPU fine-tuning is SLOW. Expect several hours per seed even
with a small model and few epochs. Start this and check back later.
"""

import string
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from noise_injection import inject_random_noise, verify_noise_injection

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cpu":
    print("WARNING: No GPU detected. Fine-tuning will be slow. Consider")
    print("reducing MAX_SEEDS or NUM_EPOCHS below if this is impractical.")


# ------------------------------------------------------------
# LOADER -- same preprocessing philosophy as prior scripts, but
# DistilBERT wants raw-ish text, not heavily stripped tokens.
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
    df['reviews_tokens_for_labeling'] = df['reviews_clean'].apply(word_tokenize)
    stop = set(stopwords.words('english'))
    df['reviews_tokens_for_labeling'] = df['reviews_tokens_for_labeling'].apply(
        lambda x: [w for w in x if w.lower() not in stop]
    )
    return df


def senti_pol_fixed(tokens):
    return TextBlob(" ".join(tokens)).sentiment.polarity


def assign_labels(df, senti_pol_fn):
    df = df.copy()
    df['senti_polarity'] = df['reviews_tokens_for_labeling'].apply(senti_pol_fn)
    condition = [
        df['senti_polarity'] > 0.05,
        (df['senti_polarity'] <= 0.05) & (df['senti_polarity'] > -0.05),
        df['senti_polarity'] <= -0.05
    ]
    df['sentiment'] = np.select(condition, ['positive', 'neutral', 'negative'], default='neutral')
    # DistilBERT gets the ORIGINAL, lightly-cleaned text (it has its own
    # tokenizer and benefits from natural sentence structure, unlike TF-IDF)
    df['reviews_text_for_bert'] = df['reviews_clean']
    return df


# ------------------------------------------------------------
# PYTORCH DATASET
# ------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True,
                                    max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ------------------------------------------------------------
# FINE-TUNING LOOP
# ------------------------------------------------------------

def fine_tune_and_evaluate(X_train_text, y_train_int, X_test_text, y_test_int,
                            num_classes, class_weights, num_epochs=2, batch_size=16):
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=num_classes
    ).to(DEVICE)

    train_dataset = ReviewDataset(X_train_text, y_train_int, tokenizer)
    test_dataset = ReviewDataset(X_test_text, y_test_int, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    )

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"    Epoch {epoch+1}/{num_epochs}, avg loss: {epoch_loss/len(train_loader):.4f}")

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return np.array(all_preds)


# ------------------------------------------------------------
# SINGLE-SEED RUN
# ------------------------------------------------------------

def run_single_seed(df_labeled, target_class, seed, noise_rates=(0.0, 0.10, 0.25, 0.40)):
    y_clean = df_labeled['sentiment']
    other_classes = [c for c in y_clean.unique() if c != target_class]
    classes_sorted = sorted(y_clean.unique())
    label_map = {c: i for i, c in enumerate(classes_sorted)}
    target_idx = label_map[target_class]

    text_col = df_labeled['reviews_text_for_bert']

    X_train_text, X_test_text, y_train_clean, y_test_clean = train_test_split(
        text_col, y_clean, test_size=0.20, random_state=seed, stratify=y_clean
    )

    seed_results = []
    for noise_rate in noise_rates:
        if noise_rate == 0.0:
            y_train_run = y_train_clean.copy()
        else:
            y_train_run, _ = inject_random_noise(
                y_train_clean, noise_rate, target_class, other_classes, random_state=seed
            )
            verify_noise_injection(y_train_clean, y_train_run, target_class, noise_rate)

        y_train_int = y_train_run.map(label_map).values
        y_test_int = y_test_clean.map(label_map).values

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(len(classes_sorted)), y=y_train_int
        )

        print(f"  [Seed {seed}, noise={noise_rate:.0%}] Fine-tuning DistilBERT...")
        start = time.time()
        preds = fine_tune_and_evaluate(
            X_train_text, y_train_int, X_test_text, y_test_int,
            num_classes=len(classes_sorted), class_weights=class_weights
        )
        print(f"    Done in {time.time()-start:.1f}s")

        report = classification_report(
            y_test_int, preds, labels=list(range(len(classes_sorted))),
            target_names=classes_sorted, output_dict=True, zero_division=0
        )
        seed_results.append({
            "Seed": seed, "Noise Rate": noise_rate,
            "Target Recall": report[target_class]["recall"],
            "Target Precision": report[target_class]["precision"],
            "Target F1": report[target_class]["f1-score"],
        })

    return pd.DataFrame(seed_results)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    SEEDS = [1, 2, 3]  # fewer seeds than prior scripts -- fine-tuning is much slower
    TARGET_CLASS = "negative"

    print("Step 1: Loading and preprocessing dataset...")
    df_raw = load_and_preprocess_amazon("data.csv")
    print("\nStep 2: Assigning clean sentiment labels...")
    df_labeled = assign_labels(df_raw, senti_pol_fixed)

    all_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        seed_df = run_single_seed(df_labeled, TARGET_CLASS, seed)
        all_results.append(seed_df)

    full_results = pd.concat(all_results, ignore_index=True)
    full_results.to_csv("finetuned_distilbert_results.csv", index=False)
    print("\nSaved results to finetuned_distilbert_results.csv")
    print("\n--- Summary: does fine-tuned DistilBERT recall degrade with noise? ---")
    summary = full_results.groupby("Noise Rate")[["Target Recall", "Target Precision", "Target F1"]].agg(['mean', 'std'])
    print(summary)


if __name__ == "__main__":
    main()
