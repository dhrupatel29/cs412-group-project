"""Test-time missingness stress test."""
import numpy as np

MASK_SEED = 2026


def mask_inputs(X, rate, seed=MASK_SEED):
    """Set each cell to NaN independently with probability `rate`.

    One uniform draw per cell with a fixed seed, so masks are nested
    (the 30% mask contains the 20% mask) and identical across models.
    """
    if rate <= 0:
        return X.copy()
    u = np.random.default_rng(seed).random(X.shape)
    return X.mask(u < rate)
