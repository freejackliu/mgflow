"""Extract valid alloy compositions from generator output vectors.

Follows the same logic as the MATLAB scripts:
  transfer_fake_to_composition_elemental.m
  transfer_fake_to_composition_elemental_1D.m
  transfer_fake_to_composition_gene.m
"""

import numpy as np
from ..data.loader import ELEMENT_SYMBOLS, composition_vector


def extract_composition(
    scores: np.ndarray,
    elements: list,
    threshold: float = 0.01,
) -> dict:
    """Convert generator scores to a valid alloy composition.

    Steps (matching MATLAB:
      1. Sort elements by score descending
      2. Normalize so fractions sum to 1
      3. Drop elements with fraction < threshold
      4. Renormalize remaining fractions to sum to 100%

    Parameters
    ----------
    scores : ndarray, shape (n_elements,)
        Generator output scores in [0, 1].
    elements : list of str
        Element symbols corresponding to each score.
    threshold : float
        Minimum fraction (in [0, 1]) to keep (default 0.01 = 1%).

    Returns
    -------
    composition : dict
        {symbol: fraction_in_percent} for surviving elements,
        sorted descending by fraction.  Empty dict if nothing survives.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = len(elements)
    if len(scores) != n:
        raise ValueError(
            f"scores length ({len(scores)}) != elements length ({n})"
        )

    # 1) Sort descending
    idx = np.argsort(scores)[::-1]

    # 2) Normalize to sum-to-1
    total = scores.sum()
    if total <= 0:
        return {}
    fracs = scores / total

    # 3) Threshold
    mask = fracs[idx] >= threshold
    keep_idx = idx[mask]

    if len(keep_idx) == 0:
        return {}

    # 4) Renormalize the survivors
    fracs_keep = fracs[keep_idx]
    fracs_keep = fracs_keep / fracs_keep.sum()

    # Build result dict (sorted descending)
    result = {}
    for i in keep_idx:
        symbol = elements[i]
        frac_pct = round(float(fracs_keep[np.where(keep_idx == i)[0][0]]) * 100, 2)
        result[symbol] = frac_pct

    return result


def extract_composition_batch(
    scores_batch: np.ndarray,
    elements: list,
    threshold: float = 0.01,
    max_elements: int = None,
) -> list:
    """Extract compositions for a batch of generator outputs.

    Parameters
    ----------
    scores_batch : ndarray, shape (batch_size, n_elements)
    elements : list of str
    threshold : float
    max_elements : int, optional
        If set, keeps only the top `max_elements` by score before thresholding.
        Useful for large element pools to avoid unrealistically multi-element compositions.

    Returns
    -------
    compositions : list of dict
    """
    compositions = []
    for i in range(len(scores_batch)):
        scores = np.asarray(scores_batch[i], dtype=np.float64).ravel()
        if max_elements is not None and max_elements < len(scores):
            # Keep only top-N scores, zero out the rest
            top_idx = np.argsort(scores)[::-1][:max_elements]
            filtered = np.zeros_like(scores)
            filtered[top_idx] = scores[top_idx]
            comp = extract_composition(filtered, elements, threshold)
        else:
            comp = extract_composition(scores, elements, threshold)
        if comp:
            compositions.append(comp)
    return compositions


def composition_to_dict(comp: dict) -> dict:
    """Convert percent-based composition dict to fraction-based for mgflow.

    mgflow expects fractions in [0, 1], e.g. {"Zr": 0.5, "Cu": 0.3}.
    """
    return {k: v / 100.0 for k, v in comp.items()}


def format_composition(comp: dict) -> str:
    """Format composition dict as a human-readable string, e.g. 'Zr50Cu30Ni20'."""
    parts = []
    for sym, frac in comp.items():
        if frac >= 0.01:
            parts.append(f"{sym}{frac:.1f}")
    return "".join(parts)
