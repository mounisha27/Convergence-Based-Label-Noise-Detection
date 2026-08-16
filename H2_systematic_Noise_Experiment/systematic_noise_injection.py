"""
Systematic Noise Injection — H2
=================================
Unlike inject_random_noise (which selects examples to corrupt uniformly
at random), this selects examples DETERMINISTICALLY — shortest reviews
in the target class first — echoing how the original first-token bug
(Paper 1) systematically damaged short reviews more than long ones
(less remaining text to outweigh a misleading first word).

Only ONE thing changes vs. inject_random_noise: how examples are
SELECTED for corruption. The relabeling target remains random, so this
isolates exactly the variable H2 is testing.
"""

import numpy as np
import pandas as pd


def inject_systematic_noise(y, X_text, noise_rate, target_class, other_classes, random_state=42):
    """
    y: pandas Series of labels
    X_text: pandas Series of review text, SAME INDEX as y
    noise_rate: float in [0,1] — fraction of target_class to corrupt
    target_class: class to inject noise into
    other_classes: possible relabel targets

    Returns: (y_noisy, corrupt_indices)
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()

    target_indices = y[y == target_class].index
    n_to_corrupt = int(round(len(target_indices) * noise_rate))

    # Deterministic selection: shortest reviews (by word count) corrupted first
    lengths = X_text.loc[target_indices].apply(lambda t: len(str(t).split()))
    sorted_indices = lengths.sort_values(ascending=True).index
    corrupt_indices = sorted_indices[:n_to_corrupt].to_numpy()

    # Relabeling target stays random — only SELECTION is systematic
    new_labels = rng.choice(other_classes, size=n_to_corrupt)
    y_noisy.loc[corrupt_indices] = new_labels

    return y_noisy, corrupt_indices


def verify_systematic_injection(y_original, y_noisy, X_text, target_class, expected_rate):
    """Same discipline as verify_noise_injection — never trust silently."""
    original_count = (y_original == target_class).sum()
    n_flipped = (y_original != y_noisy).sum()
    actual_rate = n_flipped / original_count

    # Confirm the corrupted examples really ARE shorter than the ones spared
    flipped_mask = y_original != y_noisy
    still_target_mask = (y_original == target_class) & ~flipped_mask

    flipped_lengths = X_text[flipped_mask].apply(lambda t: len(str(t).split()))
    spared_lengths = X_text[still_target_mask].apply(lambda t: len(str(t).split()))

    print(f"Target class '{target_class}':")
    print(f"  Original count: {original_count}")
    print(f"  Examples flipped: {n_flipped}")
    print(f"  Requested rate: {expected_rate:.2%}  |  Actual rate: {actual_rate:.2%}")
    print(f"  Mean word count of CORRUPTED examples: {flipped_lengths.mean():.1f}")
    print(f"  Mean word count of SPARED examples:    {spared_lengths.mean():.1f}")

    assert abs(actual_rate - expected_rate) < 0.01, "Noise rate mismatch"
    assert flipped_lengths.mean() < spared_lengths.mean(), \
        "Corrupted examples should be shorter on average — selection logic may be broken"
    print("  Verification PASSED (rate correct, corrupted examples confirmed shorter)\n")


if __name__ == "__main__":
    # Larger synthetic test — more realistic than the 5-example hand check
    np.random.seed(0)
    lengths = np.random.randint(1, 50, size=200)
    texts = pd.Series([" ".join(["word"] * n) for n in lengths])
    labels = pd.Series(["negative"] * 200)

    y_noisy, corrupted = inject_systematic_noise(
        labels, texts, noise_rate=0.25, target_class="negative",
        other_classes=["neutral", "positive"], random_state=42
    )
    verify_systematic_injection(labels, y_noisy, texts, "negative", expected_rate=0.25)
