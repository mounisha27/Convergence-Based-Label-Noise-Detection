# Convergence-Based Label Noise Detection

Code and results for a case study testing whether disagreement between resampling techniques (Coefficient of Variation, or CV, across
methods) reliably tracks injected label noise, across different noise types and target-class sizes, in a text sentiment classification task.

This repository accompanies a manuscript currently under review, extending an earlier case study from a single observed instance to a
systematically tested pattern.

## Dataset

This project uses the **Datafiniti Amazon Product Consumer Reviews Dataset (May 2019 release)**, 34,660 reviews with 20+ structured attributes.

The raw dataset is **not included in this repository** (see licensing note below). To reproduce these experiments:

1. Download the dataset from Kaggle: [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
2. Place the downloaded CSV inside each experiment folder you want to
   run, and rename it to `data.csv`

## Repository Structure

Each hypothesis has its own self-contained folder, including its own copies of the shared noise-injection utilities and its own result CSVs.

### `H1_Random_Noise_experiment_H3a/`
Random noise injected into the **negative** (minority) class.
- `noise_balancing_experiment_v2.py` — main experiment script
- `noise_injection.py` — shared utility: injects random label noise
- `noise_balancing_results.csv` — full per-method results
- `convergence_scores_all_variants.csv` — computed CV scores (3 variants)

### `H2_systematic_Noise_Experiment/`
Systematic noise injected into the negative class, compared against
H1's random noise.
- `noise_balancing_experiment_negative_class.py` — main experiment script
- `systematic_noise_injection.py` — shared utility: injects systematic
  (deterministic, shortest-reviews-first) label noise
- `h2_systematic_noise_results.csv` — full per-method results
- `h2_systematic_convergence_cv.csv` — computed CV scores

### `H3b_Noise_experiment_Positive_Class/`
Random noise injected into the **positive** (majority) class, testing
generalization across class size.
- `H3_Positive_Class_test.py` — main experiment script
- `noise_injection.py` — shared utility: injects random label noise
- `h3_positive_noise_results.csv` — full per-method results
- `h3_positive_convergence_cv.csv` — computed CV scores

### `H3c_Noise_experiment_Neutral_Class/`
Random noise injected into the **neutral** (middle) class, completing
the H3 class-size sweep.
- `H3_Neutral_Class_test.py` — main experiment script
- `noise_injection.py` — shared utility: injects random label noise
- `h3c_neutral_noise_results.csv` — full per-method results
- `h3c_neutral_convergence_cv.csv` — computed CV scores

Each `*_test.py` / `noise_balancing_experiment_*.py` script is
self-contained: it loads and preprocesses the dataset, injects noise at four rates (0%, 10%, 25%, 40%), runs the full balancing-method grid
(None, ROS, RUS, SMOTE, ADASYN, Borderline-SMOTE) across two models (SVM, Neural Network), and saves both the full per-method results and
the computed CV convergence scores to CSV.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk textblob matplotlib
```

## Running the Experiments

Each experiment is run from inside its own folder, with `data.csv` placed there first:

```bash
cd H1_Random_Noise_experiment_H3a
python noise_balancing_experiment_v2.py

cd ../H2_systematic_Noise_Experiment
python noise_balancing_experiment_negative_class.py

cd ../H3b_Noise_experiment_Positive_Class
python H3_Positive_Class_test.py

cd ../H3c_Noise_experiment_Neutral_Class
python H3_Neutral_Class_test.py
```

Each script prints progress to the console (dataset loading, deduplication, noise verification, per-model fitting progress) and
saves its result CSVs into the same folder.

## License

The code in this repository is released under the MIT License. The Amazon review dataset itself is distributed separately by Datafiniti via Kaggle under its own license terms; this repository does not redistribute the raw data. the `LICENSE` file for details. The Amazon review dataset itself is distributed separately by Datafiniti via Kaggle under its own license terms; this repository does not redistribute the raw data.
