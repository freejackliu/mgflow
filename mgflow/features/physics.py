"""Compute 14 physicochemical features — average and deviation of elemental properties.

Based on `calculatefeatures.m`.
"""

import numpy as np
from ..data.loader import (
    get_property, get_mixing_enthalpy, get_orthogonalization_matrix
)
from .ye import ye_features


def physics_features(elements, fractions) -> np.ndarray:
    """Compute 14 physicochemical features.

    Parameters
    ----------
    elements : ndarray of ints, shape (n,)
    fractions : ndarray of floats, shape (n,)

    Returns
    -------
    features : ndarray, shape (14,)
        [r_mean, delta, TM, DTM, ME, DME, Sid,
         Mean_elecnega, D_elecnega, MVEC, D_VEC,
         B_ave, D_Bulk, Eig]
    """
    n = len(elements)
    atomic_size = get_property("atomic_size")
    tm_prop = get_property("Tm")
    electroneg = get_property("electronegativity")
    vec_prop = get_property("VEC")
    bulk = get_property("Bulk_modulus")
    enthalpy = get_mixing_enthalpy()
    ortho = get_orthogonalization_matrix()  # (9, 9)

    c = fractions / fractions.sum()
    r = atomic_size[elements]
    Tm = tm_prop[elements]
    en = electroneg[elements]
    vec = vec_prop[elements]
    K = bulk[elements]

    # === 1. Mean atomic radius ===
    r_mean = np.average(r, weights=c)

    # === 2. Atomic size deviation ===
    delta = np.sqrt(np.average((1 - r / r_mean) ** 2, weights=c))

    # === 3. Mean melting temperature ===
    TM = np.average(Tm, weights=c)

    # === 4. Melting temperature deviation ===
    DTM = np.sqrt(np.average((Tm - TM) ** 2, weights=c))

    # === 5. Mean mixing enthalpy (upper-triangle sum, concentration-weighted) ===
    # MATLAB: for i=1:n-1, for j=i+1:n: ME += c(i)*c(j)*h(comp(i),comp(j))
    ME = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            h = enthalpy[elements[i], elements[j]]
            if not np.isnan(h):
                ME += c[i] * c[j] * h
    # ME is the concentration-weighted sum over unique pairs (no factor of 4)

    # === 6. Mixing enthalpy deviation ===
    DME = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            h = enthalpy[elements[i], elements[j]]
            if not np.isnan(h):
                DME += c[i] * c[j] * (h - ME) ** 2
    DME = np.sqrt(DME)

    # === 7. Ideal mixing entropy ===
    Sid = -np.sum(c * np.log(c))

    # === 8-9. Mean and deviation of electronegativity ===
    Mean_elecnega = np.average(en, weights=c)
    D_elecnega = np.sqrt(np.average((en - Mean_elecnega) ** 2, weights=c))

    # === 10-11. Mean and deviation of VEC ===
    MVEC = np.average(vec, weights=c)
    D_VEC = np.sqrt(np.average((vec - MVEC) ** 2, weights=c))

    # === 12-13. Mean and deviation of bulk modulus ===
    B_ave = np.average(K, weights=c)
    D_Bulk = np.sqrt(np.average((K - B_ave) ** 2, weights=c))

    # === 14. Minimum eigenvalue of Hessian (thermodynamic stability) ===
    # Boltzmann constant J/K, Gas constant J/mol/K
    k_B = 1.380649e-23
    R = 8.3144598

    T1 = TM * 0.001  # kJ scaling

    # Hessian matrix of size (n-1) x (n-1)
    if n >= 2:
        Hmix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    h = enthalpy[elements[i], elements[j]]
                    if not np.isnan(h):
                        Hmix[i, j] = h

        Hessian = np.zeros((n - 1, n - 1))
        for i in range(n - 1):
            Hessian[i, i] = (c[i] + c[n - 1]) * R * T1 / (c[i] * c[n - 1]) \
                            - 8 * Hmix[i, n - 1]
            for j in range(i + 1, n - 1):
                if Hmix[i, j] == 0:
                    Hmix[i, j] = 0
                Hmix[j, i] = Hmix[i, j]
                Hmix[n - 1, i] = Hmix[i, n - 1]
                Hmix[n - 1, j] = Hmix[j, n - 1]
                Hessian[i, j] = 4 * (Hmix[i, j] - Hmix[i, n - 1] - Hmix[j, n - 1]) \
                                + R * T1 / c[n - 1]
                Hessian[j, i] = Hessian[i, j]

        # Orthogonal projection
        sub_ortho = ortho[:n - 1, :n - 1]
        Ortho_Hessian = sub_ortho.T @ Hessian @ sub_ortho

        eigenvalues = np.linalg.eigvalsh(Ortho_Hessian)
        Eig = eigenvalues.min()
    else:
        Eig = 0.0  # Single element: no Hessian

    return np.array([
        r_mean, delta, TM, DTM, ME, DME, Sid,
        Mean_elecnega, D_elecnega, MVEC, D_VEC,
        B_ave, D_Bulk, Eig
    ])
