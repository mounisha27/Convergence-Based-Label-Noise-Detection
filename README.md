# Convergence-Based Label Noise Detection

Code and results for a case study testing whether disagreement between resampling techniques (Coefficient of Variation, or CV, across
methods) reliably tracks injected label noise, across different noise types and target-class sizes, in a text sentiment classification task.

This repository accompanies a manuscript currently under review, extending an earlier case study from a single observed instance to a
systematically tested, statistically validated pattern.

## Dataset

This project uses the **Datafiniti Amazon Product Consumer Reviews Dataset (May 2019 release)**, 34,660 reviews with 20+ structured attributes.

The raw dataset is **not included in this repository** (see licensing note below). To reproduce these experiments:

1. Download the dataset from Kaggle: [Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
2. Place the downloaded CSV inside each experiment folder you want to run, and rename it to `data.csv`

## Repository Structure

Each hypothesis has an original single-run experiment folder, and, where completed, a corresponding multi-seed validation folder confirming the result is statistically robust rather than an artifact of one particular data split.

### Original single-run experiments

**`H1_Random_Noise_experiment_H3a/`** — Random noise injected into the negative (minority) class.

**`H2_systematic_Noise_Experiment/`** — Systematic noise injected into the negative class, compared against H1's random noise.

**`H3b_Noise_experiment_Positive_Class/`** — Random noise injected into the positive (majority) class, testing generalization across class size.

**`H3c_Noise_experiment_Neutral_Class/`** — Random noise injected into the neutral (middle) class, completing the H3 class-size sweep.

**`H4_Cleanlab_Benchmark/`** — Controlled comparison between the CV convergence signal and confident learning (`cleanlab`), with ground-truth precision/recall scoring since noise is injected by us.

### Multi-seed statistical validation

**`H1_Multiseed_Validation/`** — H1 repeated across 10 independent random seeds, with 95% confidence intervals computed for CV at each noise rate. Confirms the core convergence-tracks-noise finding is statistically robust for both SVM and NN.

**`H2_Multiseed_validation_systematic_noise/`** — H2 repeated across 6 seeds (reusing H1's first 6 seeds, isolating noise type as the sole variable). Also finds systematic noise produces measurably wider, less stable confidence intervals than random noise, particularly for NN.

**`H3_multiseed_validation/`** — H3b and H3c each repeated across 5 seeds. Confirms 5 of 6 model-by-class-size combinations are statistically significant; the neural network's neutral-class condition is the one exception, not statistically distinguishable from no effect at 5 seeds, a genuine architecture-specific finding reported directly rather than smoothed over.

**`H4_multiseed_validation/`** — The confident learning comparison repeated across 5 seeds, confirming both CV and confident learning's precision increase with noise in a statistically robust way, while recall remains flat.

Each script prints progress to the console (dataset loading, deduplication, noise verification, per-model fitting progress) and saves both full per-seed results and computed confidence intervals to CSV.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk textblob matplotlib scipy cleanlab
```

(`cleanlab` is only required for the H4 folders.)

## Running the Experiments

Each experiment is run from inside its own folder, with `data.csv` placed there first:

```bash
# Original single-run experiments
cd H1_Random_Noise_experiment_H3a && python noise_balancing_experiment_v2.py
cd ../H2_systematic_Noise_Experiment && python noise_balancing_experiment_negative_class.py
cd ../H3b_Noise_experiment_Positive_Class && python H3_Positive_Class_test.py
cd ../H3c_Noise_experiment_Neutral_Class && python H3_Neutral_Class_test.py
cd ../H4_Cleanlab_Benchmark && python cleanlab_vs_cv_benchmark.py

# Multi-seed statistical validation (each takes several hours to run in full)
cd ../H1_Multiseed_Validation && python h1_multiseed_validation.py
cd ../H2_Multiseed_validation_systematic_noise && python h2_multiseed_validation.py
cd ../H3_multiseed_validation && python h3_multiseed_validation.py
cd ../H4_multiseed_validation && python h4_multiseed_validation.py
```

**Note on runtime:** the multi-seed validation scripts repeat the full experimental grid 5–10 times each and can take several hours to run on a standard machine without GPU acceleration. `H3_multiseed_validation.py` runs both the positive- and neutral-class sweeps sequentially in one script and is the longest, typically taking the better part of a day.

## License

The code in this repository is released under the MIT License — see the `LICENSE` file for details. The Amazon review dataset itself is distributed separately by Datafiniti via Kaggle under its own license terms; this repository does not redistribute the raw data.
