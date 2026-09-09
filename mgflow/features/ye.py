"""Compute Ye features — RMS atomic strain and stored elastic energy.

Based on `feature_ye.m`.
"""

import numpy as np
from ..data.loader import get_property


def ye_features(elements, fractions) -> np.ndarray:
    """Compute RMS strain and dimensionless stored elastic energy.

    Parameters
    ----------
    elements : ndarray of ints, shape (n,)
    fractions : ndarray of floats, shape (n,)

    Returns
    -------
    features : ndarray, shape (2,)
        [RMS_strain, elastic_energy]
    """
    n = len(elements)
    atomic_size = get_property("atomic_size")
    bulk = get_property("Bulk_modulus")

    c = fractions / fractions.sum()
    r = atomic_size[elements]      # (n,)
    K = bulk[elements]             # (n,)

    # w(i,j) and A(i,j) matrices
    w = np.zeros((n, n))
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = r[i] / r[j]
            denom = r[i] + r[j]
            w[i, j] = 2 * np.pi * (1 - np.sqrt(r[i] * (r[i] + 2 * r[j])) / denom)
            A[i, j] = 2 * np.pi * x / ((x + 1) ** 2 * np.sqrt(x * (x + 2)))

    # Packing fraction eta
    eta = 0.0
    for i in range(n):
        for j in range(n):
            x = r[i] / r[j]
            eta += 0.5 * c[j] * c[i] * (1 - np.sqrt(x * (x + 2)) / (x + 1))

    AC = A.T @ c  # (n,)
    WC = w.T @ c  # (n,)

    strain = WC / AC - 4 * np.pi * eta / AC  # (n,)

    # C matrix (diagonal with c)
    C_mat = np.diag(c)

    RMS = np.sqrt(strain @ C_mat @ strain)

    # M matrix (scaled by atomic volume and bulk modulus)
    r_mean = np.average(r, weights=c)
    K_mean = np.average(K, weights=c)
    M_mat = np.diag((r / r_mean) ** 3 * (K / K_mean) * c)
    ue = strain @ M_mat @ strain

    return np.array([RMS, ue])
