"""
Visualization Toolkit — Reusable Charts for the Whole Project
================================================================
Point each function at the relevant CSV file(s) you already have saved.
Every chart is saved as a .png file you can drop straight into the paper
or look at instead of scrolling raw CSV rows.

Covers:
  1. Balancing technique comparison (Paper 1 style — macro F1 by method)
  2. Convergence CV vs noise rate (the core H1 result)
  3. Class-size comparison across H3a/H3b/H3c (the "does this generalize" chart)
  4. Class distribution shift across noise levels (for H3b/H3c specifically)
"""

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Balancing technique comparison (Paper 1 style)
# ------------------------------------------------------------

def plot_balancing_comparison(csv_path, metric="Macro F1", output_path="balancing_comparison.png"):
    """
    Bar chart: one group of bars per balancing method, split by Model.
    Works on any of your Paper 1 -style results CSVs
    (columns: Model, Balancing, plus whatever metric you pick).
    """
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="Balancing", columns="Model", values=metric)

    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=30)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Balancing Method")
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 2. Convergence CV vs noise rate (the core H1 chart)
# ------------------------------------------------------------

def plot_convergence_vs_noise(csv_path, output_path="convergence_vs_noise.png",
                               metric_col="CV", variant_filter=None):
    """
    Line chart: noise rate on x-axis, convergence metric on y-axis,
    one line per model. Works on convergence_scores_all_variants.csv
    or any of the h3*_convergence_cv.csv files.
    """
    df = pd.read_csv(csv_path)

    if variant_filter is not None and "Variant" in df.columns:
        df = df[df["Variant"] == variant_filter]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df["Model"].unique():
        sub = df[df["Model"] == model].sort_values("Noise Rate")
        ax.plot(sub["Noise Rate"] * 100, sub[metric_col], marker="o", label=model, linewidth=2)

    ax.set_xlabel("Injected Noise Rate (%)")
    ax.set_ylabel(metric_col)
    ax.set_title(f"{metric_col} vs. Noise Rate")
    ax.legend(title="Model")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 3. Class-size comparison — H3a vs H3b vs H3c on one chart
# ------------------------------------------------------------

def plot_class_size_comparison(
    h3a_csv, h3b_csv, h3c_csv,
    labels=("H3a: Negative (minority)", "H3b: Positive (majority)", "H3c: Neutral (middle)"),
    model="NN",
    output_path="h3_class_size_comparison.png"
):
    """
    The key "does this generalize across class size" chart — three lines,
    one per target class, showing CV vs noise rate, for one model at a time.
    Run once with model='NN', once with model='SVM'.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for csv_path, label in zip([h3a_csv, h3b_csv, h3c_csv], labels):
        df = pd.read_csv(csv_path)
        sub = df[df["Model"] == model].sort_values("Noise Rate")
        ax.plot(sub["Noise Rate"] * 100, sub["CV"], marker="o", label=label, linewidth=2)

    ax.set_xlabel("Injected Noise Rate (%)")
    ax.set_ylabel("Coefficient of Variation (CV)")
    ax.set_title(f"Convergence vs. Noise Rate Across Class Sizes ({model})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# 4. Class distribution shift across noise levels (H3b/H3c)
# ------------------------------------------------------------

def plot_class_distribution_shift(noise_levels, distributions, output_path="class_shift.png"):
    """
    Stacked bar chart showing how class proportions shift as noise increases.
    `distributions` is a list of dicts, one per noise level, e.g.:
      [{'positive': 24218, 'neutral': 2621, 'negative': 889}, ...]
    matching the order of `noise_levels` e.g. [0.0, 0.10, 0.25, 0.40]
    """
    df = pd.DataFrame(distributions, index=[f"{n:.0%}" for n in noise_levels])
    ax = df.plot(kind="bar", stacked=True, figsize=(8, 5),
                  color=["#e74c3c", "#f1c40f", "#2ecc71"])
    ax.set_xlabel("Injected Noise Rate")
    ax.set_ylabel("Number of Training Examples")
    ax.set_title("Training Class Distribution Across Noise Levels")
    ax.legend(title="Class")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


# ------------------------------------------------------------
# EXAMPLE USAGE — edit paths to match your actual saved files
# ------------------------------------------------------------

if __name__ == "__main__":
    # Example 1: Paper 1 style balancing comparison
    # plot_balancing_comparison("balancing_model_results.csv", metric="Macro F1")

    # Example 2: H1's core convergence chart (CV variant only)
    # plot_convergence_vs_noise(
    #     "convergence_scores_all_variants.csv",
    #     metric_col="CV",
    #     variant_filter="3_rus_excluded_CV"
    # )

    # Example 3: H3a/b/c class-size comparison (once all three exist)
    # plot_class_size_comparison(
    #     "convergence_scores_all_variants.csv",  # H3a = negative, from H1
    #     "h3_positive_convergence_cv.csv",         # H3b = positive
    #     "h3c_neutral_convergence_cv.csv",         # H3c = neutral
    #     model="NN"
    # )

    # Example 4: class distribution shift for H3b (positive noise)
    plot_class_distribution_shift(
        noise_levels=[0.0, 0.10, 0.25, 0.40],
        distributions=[
            {'positive': 24218, 'neutral': 2621, 'negative': 889},
            {'positive': 21796, 'neutral': 3805, 'negative': 2127},
            {'positive': 18164, 'neutral': 5688, 'negative': 3876},
            {'positive': 14531, 'neutral': 7464, 'negative': 5732},  # estimated for 40%
        ],
        output_path="h1_class_shift.png"
    )
    print("\nEdit the paths above (or in the __main__ block) to point at your actual CSVs.")
