"""Compute Hu features — cohesive energy deviation and interaction energy ratio.

Based on `feature_Hu_2.m`.
"""

import numpy as np
from ..data.loader import get_interaction_energy


def hu_features(elements, fractions) -> np.ndarray:
    """Compute DcoE and MinE features.

    Parameters
    ----------
    elements : ndarray of ints, shape (n,)
    fractions : ndarray of floats, shape (n,)

    Returns
    -------
    features : ndarray, shape (2,)
        [DcoE, MinE]
    """
    n = len(elements)
    E = get_interaction_energy()  # (n_max, n_max)

    c = fractions / fractions.sum()

    # Self-interaction energies (cohesive energies)
    coE = np.array([E[elem, elem] for elem in elements])

    # Mean cohesive energy
    McoE = np.average(coE, weights=c)

    # Standard deviation of cohesive energy
    DcoE = np.sqrt(np.average((coE - McoE) ** 2, weights=c))

    # Cross-interaction energy (normalized)
    MinE_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                eij = E[elements[i], elements[j]]
                if not np.isnan(eij):
                    MinE_sum += c[i] * c[j] * eij

    MinE = MinE_sum / McoE

    return np.array([DcoE, MinE])
