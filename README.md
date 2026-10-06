# Convergence-Based Label Noise Detection

Code and results for a case study testing whether disagreement between resampling techniques (Coefficient of Variation, or CV, across
methods) reliably tracks injected label noise, across different noise types and target-class sizes, in a text sentiment classification task.

This repository accompanies the preprint **"From One Case to a Pattern: Convergence-Based Label Noise Detection Across Noise Types and Class Sizes"**
(Roy, 2026), available at [https://doi.org/10.5281/zenodo.23158886](https://doi.org/10.5281/zenodo.23158886). It extends an earlier case study
([DOI: 10.5281/zenodo.21908043](https://doi.org/10.5281/zenodo.21908043)) from a single observed instance to a systematically tested,
statistically validated pattern.

Paper section → folder: H1 (Section 4.1), H2 (4.2), H3 (4.3), H4 (4.4), **H5 (4.5, representation generalization)**. A supplementary
fine-tuning run mentioned in Section 3.8 is in `Supplementary_Finetuned_DistilBERT/`.

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

### Representation generalization (paper Section 4.5, H5)

**`H5_Representation_Shift/`** — Tests whether the CV diagnostic depends on sparse TF-IDF features, using three feature conditions with everything else held fixed (random noise, negative class, five resampling methods — None, ROS, SMOTE, ADASYN, Borderline-SMOTE — two models, 5 seeds): (A) TF-IDF at 2,500 dimensions, (B) the same TF-IDF reduced to 384 dimensions with TruncatedSVD (isolates dimensionality), and (C) `all-MiniLM-L6-v2` sentence embeddings at their native 384 dimensions (isolates semantic density). CV is reported alongside raw standard deviation and spread (max − min), since a higher baseline recall can shrink CV even when absolute disagreement is unchanged.
- `h5_representation_shift.py` — main experiment script
- `noise_injection.py` — shared utility: injects random label noise
- `h5_representation_shift_full_results.csv` — per-seed, per-method recall (600 rows)
- `h5_representation_shift_decomposed.csv` — per-seed mean, std, spread, and CV
- `h5_ci_CV.csv`, `h5_ci_std_recall.csv`, `h5_ci_spread.csv` — 95% confidence intervals (t-distribution) for each metric
- `h5_per_resampler_breakdown.csv` — mean recall for each individual resampling method

### Supplementary: fine-tuned transformer run (not a result of the paper)

**`Supplementary_Finetuned_DistilBERT/`** — A supplementary run (disclosed in Section 3.8 of the paper) that fine-tunes DistilBERT with class-weighted loss under the same noise-injection protocol (3 seeds). Because resampling operates on fixed feature vectors and fine-tuning updates weights from raw text, the five-method comparison cannot be reproduced here, so **this run does not test the CV diagnostic itself**; it only records how a single fine-tuned classifier's target-class recall behaves under injected noise.
- `finetuned_distilbert.py` — fine-tuning script
- `noise_injection.py` — shared utility
- `finetuned_distilbert_results.csv` — recall, precision, and F1 per seed and noise rate

Each script prints progress to the console (dataset loading, deduplication, noise verification, per-model fitting progress) and saves both full per-seed results and computed confidence intervals to CSV.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk textblob matplotlib scipy cleanlab
```

(`cleanlab` is only required for the H4 folders.)

The H5 and supplementary folders additionally need:

```bash
pip install sentence-transformers transformers torch
```
(`sentence-transformers` for H5, `transformers` and `torch` for the fine-tuning run. The first run downloads the pretrained models automatically.)

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

# Representation generalization (paper Section 4.5) and supplementary fine-tuning run
cd ../H5_Representation_Shift && python h5_representation_shift.py
cd ../Supplementary_Finetuned_DistilBERT && python finetuned_distilbert.py
```

**Note on runtime:** the multi-seed validation scripts repeat the full experimental grid 5–10 times each and can take several hours to run on a standard machine without GPU acceleration. `H3_multiseed_validation.py` runs both the positive- and neutral-class sweeps sequentially in one script and is the longest, typically taking the better part of a day. All timings below are from a CPU-only personal computer (no GPU): `h5_representation_shift.py` took roughly 14 hours in total (3 feature conditions × 5 seeds), and the supplementary fine-tuning run took roughly 6 hours per noise-rate run, about 72 hours in total (3 seeds × 4 noise rates), so budget accordingly or reduce `SEEDS` at the top of each script's `main()`.

## Citation

```bibtex
@misc{roy2026convergence,
  title  = {From One Case to a Pattern: Convergence-Based Label Noise Detection Across Noise Types and Class Sizes},
  author = {Roy, Mounisha},
  year   = {2026},
  doi    = {10.5281/zenodo.23158886},
  note   = {Preprint}
}
```

## License

The code in this repository is released under the MIT License — see the `LICENSE` file for details. The Amazon review dataset itself is distributed separately by Datafiniti via Kaggle under its own license terms; this repository does not redistribute the raw data.
