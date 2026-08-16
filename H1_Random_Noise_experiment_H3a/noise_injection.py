"""
Noise Injection — Step 1: Random noise, controlled rate, class size held constant
====================================================================================
Takes the CLEAN, deduplicated, correctly-labeled dataset and injects a
controlled amount of RANDOM label noise into the minority (negative)
class, without changing how many negative examples exist.

This is deliberately the simplest possible noise type first (random,
not systematic) — get this working and verified before adding the
systematic-noise condition from the experimental design doc.
"""

import numpy as np
import pandas as pd


def inject_random_noise(y, noise_rate, target_class, other_classes, random_state=42):
    """
    Flips `noise_rate` fraction of `target_class` labels to a randomly
    chosen OTHER class. Does NOT change how many examples exist in
    target_class — only whether their label is correct.

    y: pandas Series of string labels
    noise_rate: float in [0, 1] — fraction of target_class examples to corrupt
    target_class: the class to inject noise into (e.g. "negative")
    other_classes: list of classes noise can flip TO (e.g. ["neutral", "positive"])
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()

    target_indices = y[y == target_class].index.to_numpy()
    n_to_corrupt = int(round(len(target_indices) * noise_rate))

    corrupt_indices = rng.choice(target_indices, size=n_to_corrupt, replace=False)
    new_labels = rng.choice(other_classes, size=n_to_corrupt)

    y_noisy.loc[corrupt_indices] = new_labels

    return y_noisy, corrupt_indices


def verify_noise_injection(y_original, y_noisy, target_class, expected_rate):
    """
    Sanity checks — run this every time, don't trust the injection blindly.
    """
    original_count = (y_original == target_class).sum()
    noisy_count = (y_noisy == target_class).sum()
    n_flipped = (y_original != y_noisy).sum()
    actual_rate = n_flipped / original_count

    print(f"Target class '{target_class}':")
    print(f"  Original count: {original_count}")
    print(f"  Count after noise injection (should be LOWER, some flipped away): {noisy_count}")
    print(f"  Examples flipped: {n_flipped}")
    print(f"  Requested noise rate: {expected_rate:.2%}")
    print(f"  Actual noise rate:    {actual_rate:.2%}")

    assert abs(actual_rate - expected_rate) < 0.01, "Noise rate mismatch — check the injection logic"
    print("  Verification PASSED\n")


# ------------------------------------------------------------
# QUICK SANITY TEST — run this alone first, before touching the real grid
# ------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic example so this is checkable by hand
    y_test = pd.Series(
        ["negative"] * 20 + ["neutral"] * 60 + ["positive"] * 20
    )

    print("=== Testing at 25% noise rate ===")
    y_noisy, flipped_idx = inject_random_noise(
        y_test, noise_rate=0.25, target_class="negative",
        other_classes=["neutral", "positive"], random_state=42
    )
    verify_noise_injection(y_test, y_noisy, "negative", expected_rate=0.25)

    print("=== Testing at 40% noise rate ===")
    y_noisy2, flipped_idx2 = inject_random_noise(
        y_test, noise_rate=0.40, target_class="negative",
        other_classes=["neutral", "positive"], random_state=42
    )
    verify_noise_injection(y_test, y_noisy2, "negative", expected_rate=0.40)

    # Confirm total dataset size never changes — only labels move, nothing added/removed
    assert len(y_test) == len(y_noisy) == len(y_noisy2), "Row count changed! Bug."
    print("Row count unchanged across all conditions — confirmed noise injection")
    print("only relabels, never adds or removes examples.")