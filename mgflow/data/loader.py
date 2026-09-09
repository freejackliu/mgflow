"""Load elemental property data and inter-element interaction matrices."""

import numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "elemental"


def _load_array(name: str) -> np.ndarray:
    path = _DATA_DIR / f"{name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return np.load(path)


# Lazy-loaded singletons
_props_cache = {}


def _ensure_loaded():
    if not _props_cache:
        _props_cache["atomic_size"] = _load_array("atomic_size")
        _props_cache["Tm"] = _load_array("Tm")
        _props_cache["electronegativity"] = _load_array("electronegativity")
        _props_cache["VEC"] = _load_array("VEC")
        _props_cache["Youngs_modulus"] = _load_array("Youngs_modulus")
        _props_cache["Bulk_modulus"] = _load_array("Bulk_modulus")
        _props_cache["mixing_enthalpy"] = _load_array("mixing_enthalpy")
        _props_cache["interaction_energy"] = _load_array("interaction_energy")
        _props_cache["orthogonalization_matrix"] = _load_array("orthogonalization_matrix")


def get_property(name: str) -> np.ndarray:
    """Return a 1-D array indexed by element number (1-based). Entry 0 is unused."""
    _ensure_loaded()
    return _props_cache[name]


def get_mixing_enthalpy() -> np.ndarray:
    """Return (n_max, n_max) mixing enthalpy matrix in kJ/mol."""
    _ensure_loaded()
    return _props_cache["mixing_enthalpy"]


def get_interaction_energy() -> np.ndarray:
    """Return (n_max, n_max) interaction energy matrix."""
    _ensure_loaded()
    return _props_cache["interaction_energy"]


def get_orthogonalization_matrix() -> np.ndarray:
    """Return (9, 9) orthogonalization matrix for Hessian computation."""
    _ensure_loaded()
    return _props_cache["orthogonalization_matrix"]


# Element symbol lookup (1-based, entry 0 is empty)
ELEMENT_SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
]

SYMBOL_TO_NO = {sym: i for i, sym in enumerate(ELEMENT_SYMBOLS) if sym}


# ── Training dataset loader (from NPZ, includes metadata) ──
_TRAIN_DATA_DIR = Path(__file__).parent / "datasets"


def load_dataset(prop, include_metadata=True):
    """Load a training dataset for the given property from NPZ.

    Parameters
    ----------
    prop : str
        One of 'Tg', 'Tx', 'Tl', 'E', 'H'.
    include_metadata : bool
        If True, return metadata arrays (mg, bmg, ribbon, D, base_element).

    Returns
    -------
    dict with keys:
        names      : ndarray of str, alloy names
        elem_nos   : ndarray (n, 8), element atomic numbers
        fracs      : ndarray (n, 8), atomic fractions
        labels     : ndarray (n,), target property values
        mg         : ndarray (n,), int8  — 1=MG, -1=no data  (if include_metadata)
        bmg        : ndarray (n,), int8  — 1=BMG, 0=not BMG, -1=no data
        ribbon     : ndarray (n,), int8  — 1=ribbon, 0=not ribbon, -1=no data
        D          : ndarray (n,), float — critical diameter (mm), NaN=no data
        base_element : ndarray (n,), int16 — atomic number of base element, -1=no data
    """
    path = _TRAIN_DATA_DIR / f"dataset_{prop}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path, allow_pickle=True)
    result = {
        "names": data["names"],
        "elem_nos": data["elem_nos"],
        "fracs": data["fracs"],
        "labels": data["labels"],
    }
    if include_metadata and "mg" in data:
        result["mg"] = data["mg"]
        result["bmg"] = data["bmg"]
        result["ribbon"] = data["ribbon"]
        result["D"] = data["D"]
        result["base_element"] = data["base_element"]

    return result


def list_available_datasets():
    """Return list of property names with available training datasets."""
    props = []
    for prop in ["Tg", "Tx", "Tl", "E", "H"]:
        path = _TRAIN_DATA_DIR / f"dataset_{prop}.npz"
        if path.exists():
            props.append(prop)
    return props


def composition_vector(composition):
    """Convert composition to (element_numbers, fractions) arrays.

    Parameters
    ----------
    composition : dict or list of tuples
        e.g. {"Zr": 0.5, "Cu": 0.3, "Ni": 0.2}
        or [(40, 0.5), (29, 0.3), (28, 0.2)]

    Returns
    -------
    elements : ndarray of ints, shape (n,)
    fractions : ndarray of floats, shape (n,)
    """
    if isinstance(composition, dict):
        # keys are element symbols
        items = []
        for sym, frac in composition.items():
            no = SYMBOL_TO_NO.get(sym)
            if no is None:
                raise ValueError(f"Unknown element symbol: {sym}")
            items.append((no, frac))
    else:
        items = list(composition)

    elements = np.array([int(item[0]) for item in items])
    fractions = np.array([float(item[1]) for item in items])
    fractions = fractions / fractions.sum()  # normalize
    return elements, fractions
