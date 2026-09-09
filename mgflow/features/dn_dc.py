"""Compute dn/dc features (6 values) — atomic-level strain descriptors.

Based on `dn_dc_features.m` and `cal_meandndc.m`.

NOTE: The original MATLAB code uses Symbolic Math Toolbox (functions
Mean_dc, Mean_dn, etc.) which are not available. This module provides
a numerical approximation based on the mathematical framework:
  - b: site-level corrected strain
  - c: concentrations
  - d: diagonal matrix of WC/AC (site stiffness)
  - p: interaction kernel
"""

import numpy as np
from ..data.loader import get_property


def dn_dc_features(elements, fractions) -> np.ndarray:
    """Compute 6 dn/dc features.

    Parameters
    ----------
    elements : ndarray of ints, shape (n,)
    fractions : ndarray of floats, shape (n,)

    Returns
    -------
    features : ndarray, shape (6,)
    """
    n = len(elements)
    radius = get_property("atomic_size")

    c = fractions / fractions.sum()
    r = radius[elements]

    # --- w and A matrices (same as ye_features) ---
    w = np.zeros((n, n))
    A = np.zeros((n, n))
    x = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x[i, j] = r[i] / r[j]
            denom = r[i] + r[j]
            w[i, j] = 2 * np.pi * (1 - np.sqrt(r[i] * (r[i] + 2 * r[j])) / denom)
            A[i, j] = 2 * np.pi * x[i, j] / ((x[i, j] + 1) ** 2 *
                                               np.sqrt(x[i, j] * (x[i, j] + 2)))

    # --- Packing fraction eta ---
    eta = 0.0
    for i in range(n):
        for j in range(n):
            eta += 0.5 * c[j] * c[i] * (1 - np.sqrt(x[i, j] * (x[i, j] + 2)) / (x[i, j] + 1))

    # --- AC and WC vectors ---
    AC = A.T @ c  # (n,)
    WC = w.T @ c  # (n,)

    # --- b vector (corrected site strain) ---
    b = (WC - 4 * np.pi * eta) / AC  # (n,)

    # --- d matrix (diagonal: site-level strain per concentration) ---
    d_diag = WC / AC  # (n,)
    d_mat = np.diag(d_diag)  # (n, n)

    # --- p matrix (interaction kernel) ---
    p_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p_mat[i, j] = w[i, j] / AC[i]

    # === Compute 6 features (numerical moments) ===
    # The original MATLAB uses symbolic functions (Mean_dc, Mean_dn, etc.)
    # that operate on the d (site-stiffness) and p (interaction) matrices.
    # Here we compute analogous statistics from the available variables.

    # 1. Mean_dc: concentration-weighted mean of d (diagonal of WC/AC)
    mean_dc = np.sum(c * d_diag)

    # 2. Mean_dn: simple mean of d
    mean_dn = np.mean(d_diag)

    # 3. Meansquare_dc: concentration-weighted second moment of b (corrected strain)
    #    sqrt(Σ_i c_i * b_i²) — site-strain magnitude
    meansq_dc = np.sqrt(np.sum(c * b ** 2))

    # 4. Meansquare_dn: unweighted RMS of d
    meansq_dn = np.sqrt(np.mean(d_diag ** 2))

    # 5. SD_dc: concentration-weighted std of d
    var_dc = np.sum(c * (d_diag - mean_dc) ** 2)
    sd_dc = np.sqrt(var_dc)

    # 6. SD_dn: simple std of d
    sd_dn = np.std(d_diag, ddof=0)

    return np.array([mean_dc, mean_dn, meansq_dc, meansq_dn, sd_dc, sd_dn])
