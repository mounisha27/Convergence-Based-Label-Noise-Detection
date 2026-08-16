# Convergence-Based Label Noise Detection

Code and results for a case study testing whether disagreement between resampling techniques (Coefficient of Variation, or CV, across methods) reliably 
tracks injected label noise, across different noise types and target-class sizes, in a text sentiment classification task.
This repository accompanies a manuscript currently under review, extending an earlier case study from a single observed instance to a
systematically tested pattern.

## Dataset

This project uses the **Datafiniti Amazon Product Consumer Reviews Dataset (May 2019 release)**, 34,660 reviews with 20+ structured attributes.

The raw dataset is **not included in this repository** (see licensing note below). To reproduce these experiments:

1. Download the dataset from Kaggle: [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
2. Place the downloaded CSV in this folder and rename it to `data.csv`

## Repository Structure

| File | Hypothesis | Description |
|---|---|---|
| `noise_injection.py` | — | Shared utility: injects **random** label noise, with built-in verification |
| `systematic_noise_injection.py` | — | Shared utility: injects **systematic** (deterministic, shortest-reviews-first) label noise, with built-in verification |
| `h1_random_noise_negative.py` | H1 | Random noise injected into the negative (minority) class |
| `h2_systematic_noise_negative.py` | H2 | Systematic noise injected into the negative class, compared against H1's random noise |
| `h3b_random_noise_positive.py` | H3b | Random noise injected into the positive (majority) class, testing generalization across class size |
| `h3c_random_noise_neutral.py` | H3c | Random noise injected into the neutral (middle) class, completing the H3 class-size sweep |
| `visualize_results.py` | — | Plotting utilities for all result CSVs (convergence-vs-noise charts, class-size comparisons, class-distribution shifts) |
| `check_dataset.py` | — | Quick diagnostic: confirms row count and column names after downloading `data.csv` |

Each `h*.py` script is self-contained: it loads and preprocesses the dataset, injects noise at four rates (0%, 10%, 25%, 40%), runs the
full balancing-method grid (None, ROS, RUS, SMOTE, ADASYN, Borderline-SMOTE) across two models (SVM, Neural Network), and saves
both the full per-method results and the computed CV convergence scores to CSV.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk textblob matplotlib
```

## Running the Experiments

```bash
# Confirm the dataset loaded correctly
python check_dataset.py

# Run each hypothesis (can be run independently, in any order)
python h1_random_noise_negative.py
python h2_systematic_noise_negative.py
python h3b_random_noise_positive.py
python h3c_random_noise_neutral.py
```

Each script prints progress to the console (dataset loading, deduplication, noise verification, 
per-model fitting progress) and saves two output files: `*_noise_results.csv` (full per-method results) 
and `*_convergence_cv.csv` (computed CV scores).

## License

The code in this repository is released under MIT License. The Amazon review dataset itself is distributed separately by
Datafiniti via Kaggle under its own license terms; this repository does not redistribute the raw data.
