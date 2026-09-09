"""Compute all features from a composition.

Supports two modes:
  - 25-feature set: dn_dc(6) + ye(2) + Scorr(1) + Hu(2) + physics(14)
  - 11-feature set: dn_dc(6) + ye(2) + Scorr(1) + Hu(2)
"""

import numpy as np
from ..data.loader import composition_vector
from .dn_dc import dn_dc_features
from .ye import ye_features
from .scorr import scorr_features
from .hu import hu_features
from .physics import physics_features


def compute_all_features(composition, include_physics=True) -> np.ndarray:
    """Compute all features for a given composition.

    Parameters
    ----------
    composition : dict or list of tuples
        Examples:
          {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}
          [(40, 0.5), (29, 0.3), (28, 0.2)]
    include_physics : bool
        If True (default), compute all 25 features.
        If False, compute 11 features (dn_dc + ye + Scorr + Hu).

    Returns
    -------
    features : ndarray, shape (25,) or (11,)
    """
    elements, fractions = composition_vector(composition)

    f_dndc = dn_dc_features(elements, fractions)     # 6
    f_ye = ye_features(elements, fractions)           # 2
    f_scorr = scorr_features(elements, fractions)     # 1
    f_hu = hu_features(elements, fractions)           # 2

    if include_physics:
        f_phys = physics_features(elements, fractions)  # 14
        return np.concatenate([f_dndc, f_ye, f_scorr, f_hu, f_phys])
    else:
        return np.concatenate([f_dndc, f_ye, f_scorr, f_hu])


FEATURE_NAMES_25 = [
    "dn_dc1(mean_dc)", "dn_dc2(mean_dn)", "dn_dc3(meansq_dc)",
    "dn_dc4(meansq_dn)", "dn_dc5(sd_dc)", "dn_dc6(sd_dn)",
    "ye_RMS", "ye_elastic_energy",
    "Scorr",
    "Hu_DcoE", "Hu_MinE",
    "r_mean", "delta_r", "Tm_mean", "Tm_dev",
    "H_mix_mean", "H_mix_dev", "S_id",
    "EN_mean", "EN_dev", "VEC_mean", "VEC_dev",
    "B_mean", "B_dev", "Eig_Hessian",
]

FEATURE_NAMES_11 = [
    "dn_dc1(mean_dc)", "dn_dc2(mean_dn)", "dn_dc3(meansq_dc)",
    "dn_dc4(meansq_dn)", "dn_dc5(sd_dc)", "dn_dc6(sd_dn)",
    "ye_RMS", "ye_elastic_energy",
    "Scorr",
    "Hu_DcoE", "Hu_MinE",
]


def print_features(composition):
    """Pretty-print all features for a composition."""
    from ..data.loader import ELEMENT_SYMBOLS
    elements, fractions = composition_vector(composition)
    feat = compute_all_features(composition)

    comp_str = {ELEMENT_SYMBOLS[e]: float(f) for e, f in zip(elements, fractions)}
    print(f"Composition: {comp_str}")
    print(f"  Elements: {[ELEMENT_SYMBOLS[e] for e in elements]}")
    print(f"  Fractions:    {fractions}")
    print()
    print(f"{'Feature':<25s} {'Value':>12s}")
    print("-" * 38)
    for name, val in zip(FEATURE_NAMES_25, feat):
        print(f"{name:<25s} {val:>12.6f}")
    return feat
