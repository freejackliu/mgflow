"""Compute Scorr — mixing entropy corrected by elastic and chemical fluctuations.

Based on `CalScorr.m` and `SE.m`.
"""

import numpy as np
from ..data.loader import get_property, get_mixing_enthalpy
from .ye import ye_features


# Boltzmann constant (J/K)
_K_B = 1.380649e-23
# Avogadro constant
_NA = 6.02214076e23


def _SE(x):
    """Shannon-like entropy of fluctuation parameter.

    Original: SE.m:
        if x <= 0 || x >= 1, S = 0
        else S = -x*log(x) - (1-x)*log(1-x)
    """
    if x <= 0 or x >= 1:
        return 0.0
    return -x * np.log(x) - (1 - x) * np.log(1 - x)


def scorr_features(elements, fractions) -> np.ndarray:
    """Compute Scorr parameter.

    Parameters
    ----------
    elements : ndarray of ints, shape (n,)
    fractions : ndarray of floats, shape (n,)

    Returns
    -------
    features : ndarray, shape (1,)
        [Scorr]
    """
    n = len(elements)
    atomic_size = get_property("atomic_size")
    tm_prop = get_property("Tm")
    bulk = get_property("Bulk_modulus")
    enthalpy = get_mixing_enthalpy()

    c = fractions / fractions.sum()
    r = atomic_size[elements]
    Tm = tm_prop[elements]
    K = bulk[elements]

    # Averages
    r_avg = np.average(r, weights=c)
    Tm_avg = np.average(Tm, weights=c)
    K_avg = np.average(K, weights=c)

    # Volume per atom (m^3)
    V = 4 / 3 * np.pi * (r_avg * 1e-10) ** 3  # r in Angstrom -> meters

    # Elastic energy fluctuation
    RMS = ye_features(elements, fractions)[0]
    x_e = 3 * np.sqrt(2) * RMS * np.sqrt(K_avg * V / (_K_B * Tm_avg))

    # Mixing enthalpy fluctuation
    H_sum = 0.0
    H_sq_sum = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                hij = enthalpy[elements[i], elements[j]]
                if not np.isnan(hij):
                    H_sum += c[i] * c[j] * hij

    # Average mixing enthalpy (per formula in CalScorr.m: H_ave = 4 * Σ c_i c_j h_ij)
    H_avg = 4 * H_sum

    temp = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                hij = enthalpy[elements[i], elements[j]]
                if not np.isnan(hij):
                    temp += c[i] * c[j] * (hij - H_avg / 4) ** 2
    # Actually the original code uses data2_ld which is the full enthalpy matrix
    # Let's use enthalpy directly
    temp2 = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                hij = enthalpy[elements[i], elements[j]]
                if not np.isnan(hij):
                    temp2 += c[i] * c[j] * (hij - H_sum) ** 2

    # Chemical bond fluctuation
    x_c = np.sqrt(2 * np.sqrt(temp2) * 1e3 / _NA / _K_B / Tm_avg)

    # Ideal mixing entropy
    S_id = -np.sum(c * np.log(c))

    Scorr_val = S_id + _SE(x_e + x_c)

    return np.array([Scorr_val])
