"""Dataset viewer for MGflow training data.

Provides convenient access to the parsed alloy datasets from Excel.
Each dataset contains composition + label for one target property.

Usage:
    from mgflow.dataset_viewer import show_all, query_alloy, query_element, show_stats
"""

import numpy as np
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "datasets"
_PROP_LABELS = {
    "Tg": "Tg (K)",
    "Tx": "Tx (K)",
    "Tl": "Tl (K)",
    "E":  "E (GPa)",
    "H":  "Hv (GPa)",
}

# ── Element symbol mapping (same as data_loader.py) ──
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
NO_TO_SYMBOL = {i: sym for sym, i in SYMBOL_TO_NO.items()}


class AlloyDB:
    """In-memory database of all training datasets."""

    def __init__(self):
        self._datasets = {}
        self._alloy_to_props = {}  # name -> {prop: label_value}
        self._alloy_to_meta = {}  # name -> {mg, bmg, ribbon, D, base_element}
        for prop in ["Tg", "Tx", "Tl", "E", "H"]:
            path = _DATA_DIR / f"dataset_{prop}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=True)
            names = data["names"]
            elem_nos = data["elem_nos"]
            fracs = data["fracs"]
            labels = data["labels"]
            self._datasets[prop] = {
                "names": names,
                "elem_nos": elem_nos,
                "fracs": fracs,
                "labels": labels,
            }
            for i, name in enumerate(names):
                self._alloy_to_props.setdefault(name, {})[prop] = float(labels[i])

            # Load metadata if present (only overwrite if meaningful)
            if "mg" in data:
                for i, name in enumerate(names):
                    meta = self._alloy_to_meta.setdefault(name, {})
                    mg_val = int(data["mg"][i])
                    if mg_val != -1:
                        meta["mg"] = mg_val
                    bmg_val = int(data["bmg"][i])
                    if bmg_val != -1:
                        meta["bmg"] = bmg_val
                    rib_val = int(data["ribbon"][i])
                    if rib_val != -1:
                        meta["ribbon"] = rib_val
                    d_val = float(data["D"][i])
                    if not np.isnan(d_val):
                        meta["D"] = d_val
                    base = int(data["base_element"][i])
                    if base != -1:
                        meta["base_element"] = base
                        meta["base_symbol"] = NO_TO_SYMBOL.get(base, "")

    @property
    def properties(self):
        return list(self._datasets.keys())

    def get_dataset(self, prop):
        """Return raw arrays for a property dataset."""
        return self._datasets.get(prop)

    def get_composition(self, name):
        """Return (element_symbols, fractions) for a named alloy."""
        for prop, data in self._datasets.items():
            mask = data["names"] == name
            if mask.any():
                idx = np.where(mask)[0][0]
                nos = data["elem_nos"][idx]
                fracs = data["fracs"][idx]
                mask_nz = fracs > 0
                syms = [NO_TO_SYMBOL[n] for n in nos[mask_nz]]
                fracs_nz = fracs[mask_nz]
                return syms, fracs_nz
        return None, None

    def find_alloys_by_element(self, element_symbol):
        """Find all alloys containing a given element symbol (e.g. 'Zr')."""
        z = SYMBOL_TO_NO.get(element_symbol)
        if z is None:
            return []
        results = {}
        for prop, data in self._datasets.items():
            for i, name in enumerate(data["names"]):
                nos = data["elem_nos"][i]
                if z in nos[:8]:
                    results.setdefault(name, set()).add(prop)
        return results

    def format_composition(self, name):
        """Format composition as a string like 'Zr₅₀Cu₅₀'."""
        syms, fracs = self.get_composition(name)
        if syms is None:
            return name
        parts = []
        for s, f in zip(syms, fracs):
            pct = int(round(f * 100))
            parts.append(f"{s}{pct}")
        return "".join(parts)


# ── Singleton ──
_db = None


def _get_db():
    global _db
    if _db is None:
        _db = AlloyDB()
    return _db


# ═══════════════════════════════════════════════════════════════════════════
# Public query functions
# ═══════════════════════════════════════════════════════════════════════════

def show_stats():
    """Print a summary table of all datasets."""
    db = _get_db()
    print(f"{'Property':<6s} {'Samples':>8s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'#Elements':>10s}")
    print("-" * 56)
    for prop in db.properties:
        data = db.get_dataset(prop)
        n = len(data["labels"])
        minv, maxv = data["labels"].min(), data["labels"].max()
        meanv = data["labels"].mean()
        # Count unique element occurrences
        all_nos = data["elem_nos"].flatten()
        unique_elems = len(set(all_nos[all_nos > 0]))
        print(f"{prop:<6s} {n:>8d} {minv:>10.1f} {maxv:>10.1f} {meanv:>10.1f} {unique_elems:>10d}")
    
    # Count unique alloys across all datasets
    all_names = set()
    for prop in db.properties:
        all_names.update(db.get_dataset(prop)["names"].tolist())
    print(f"\nTotal unique alloys across all datasets: {len(all_names)}")
    
    # Element frequency across all datasets
    from collections import Counter
    elem_counter = Counter()
    for prop in db.properties:
        for row_nos in db.get_dataset(prop)["elem_nos"]:
            for z in row_nos[row_nos > 0]:
                elem_counter[z] += 1
    print(f"\nTop 15 most common elements:")
    for z, cnt in elem_counter.most_common(15):
        print(f"  {NO_TO_SYMBOL[z]:>3s} (Z={z:2d}): {cnt:5d} occurrences")


def show_all(sort_by="name"):
    """Print the entire dataset as a table.

    Parameters
    ----------
    sort_by : str
        'name' to sort alphabetically, 'prop_name' to sort by property value.
    """
    db = _get_db()
    rows = []
    for prop in db.properties:
        data = db.get_dataset(prop)
        for i, name in enumerate(data["names"]):
            syms, fracs = db.get_composition(name)
            comp_str = "".join(f"{s}{int(round(f*100))}" for s, f in zip(syms, fracs))
            rows.append((name, comp_str, prop, float(data["labels"][i])))
    
    if sort_by == "name":
        rows.sort(key=lambda r: r[0])
    elif sort_by == "prop":
        rows.sort(key=lambda r: (r[2], r[3]))
    
    print(f"{'Alloy':<45s} {'Composition':<40s} {'Prop':<5s} {'Value':>10s}")
    print("-" * 102)
    for name, comp, prop, val in rows:
        print(f"{name:<45s} {comp:<40s} {prop:<5s} {val:>10.2f}")
    print(f"\nTotal: {len(rows)} data points")


def query_alloy(name_substring, verbose=True):
    """Find and display all alloys whose name contains a substring.

    Parameters
    ----------
    name_substring : str
        Case-insensitive substring to match in alloy name.
    verbose : bool
        If True, print results; always returns list of matching names.
    """
    db = _get_db()
    name_substring = name_substring.lower()
    matches = []
    for prop in db.properties:
        for name in db.get_dataset(prop)["names"]:
            if name_substring in name.lower():
                matches.append(name)
    matches = sorted(set(matches))
    
    if verbose:
        if not matches:
            print(f"No alloys found matching '{name_substring}'")
        else:
            print(f"Found {len(matches)} alloys matching '{name_substring}':")
            print("-" * 60)
            for name in matches:
                syms, fracs = db.get_composition(name)
                comp_str = "".join(f"{s}{int(round(f*100))}" for s, f in zip(syms, fracs))
                props = db._alloy_to_props.get(name, {})
                prop_str = "  |  ".join(f"{p}={props[p]:.1f}" for p in ["Tg", "Tx", "Tl", "E", "H"] if p in props)
                print(f"  {name:<45s} {comp_str:<30s}")
                if prop_str:
                    print(f"  {'':>45s} {prop_str}")
    
    return matches


def _format_meta_str(meta):
    """Format metadata line for compact display."""
    parts = []
    if meta.get("mg") == 1:
        parts.append("MG")
        if meta.get("bmg") == 1:
            parts.append("BMG")
        elif meta.get("bmg") == 0:
            parts.append("非BMG")
    if meta.get("ribbon") == 1:
        parts.append("Ribbon")
    d = meta.get("D")
    if d is not None and d > 0:
        parts.append(f"D={d:.1f}mm")
    base = meta.get("base_symbol", "")
    if base:
        parts.append(f"基{base}")
    return "  |  ".join(parts) if parts else ""


def query_element(element_symbol, verbose=True):
    """Find all alloys containing a specific element.

    Parameters
    ----------
    element_symbol : str
        Element symbol, e.g. 'Zr', 'Cu', 'Fe'.
    verbose : bool
        If True, print results.
    """
    db = _get_db()
    results = db.find_alloys_by_element(element_symbol)
    names = sorted(results.keys())
    
    if verbose:
        if not names:
            print(f"No alloys contain {element_symbol}")
        else:
            print(f"Found {len(names)} alloys containing {element_symbol}:")
            print("-" * 100)
            for name in names:
                syms, fracs = db.get_composition(name)
                comp_str = "".join(f"{s}{int(round(f*100))}" for s, f in zip(syms, fracs))
                props = db._alloy_to_props.get(name, {})
                prop_str = "  |  ".join(f"{p}={props[p]}" for p in ["Tg", "Tx", "Tl", "E", "H"] if p in props)
                meta_str = _format_meta_str(db._alloy_to_meta.get(name, {}))
                print(f"  {name:<45s} {comp_str:<30s}")
                if prop_str:
                    print(f"  {'':>45s} {prop_str}")
                if meta_str:
                    print(f"  {'':>45s} [{meta_str}]")
    
    return names


def show_composition(name):
    """Display the composition and all known properties for a given alloy."""
    db = _get_db()
    syms, fracs = db.get_composition(name)
    if syms is None:
        print(f"Alloy '{name}' not found in any dataset.")
        return
    
    comp_str = "".join(f"{s}{int(round(f*100))}" for s, f in zip(syms, fracs))
    print(f"Alloy: {name}")
    print(f"Composition: {comp_str}")
    print(f"Elements: {', '.join(f'{s} {f*100:.1f}%' for s, f in zip(syms, fracs))}")
    print(f"Number of components: {len(syms)}")
    
    # ── Metadata ──
    meta = db._alloy_to_meta.get(name, {})
    if meta:
        base_sym = meta.get("base_symbol", "")
        mg_str = {1: "Yes", 0: "No", -1: "N/A"}.get(meta.get("mg"), "N/A")
        bmg_str = {1: "Yes", 0: "No", -1: "N/A"}.get(meta.get("bmg"), "N/A")
        rib_str = {1: "Yes", 0: "No", -1: "N/A"}.get(meta.get("ribbon"), "N/A")
        d_str = f"{meta['D']:.1f} mm" if meta.get("D") is not None else "N/A"
        print(f"Base element: {base_sym}")
        print(f"Metal Glass: {mg_str}  |  BMG: {bmg_str}  |  Ribbon: {rib_str}  |  Dmax: {d_str}")
    print()
    
    props = db._alloy_to_props.get(name, {})
    if props:
        print(f"{'Property':<10s} {'Value':>12s}")
        print("-" * 24)
        for p in ["Tg", "Tx", "Tl", "E", "H"]:
            if p in props:
                label = _PROP_LABELS.get(p, p)
                print(f"{label:<10s} {props[p]:>12.2f}")
    else:
        print("No property data available.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m mgflow.dataset_viewer stats          # Show dataset statistics")
        print("  python -m mgflow.dataset_viewer all             # List all data points")
        print("  python -m mgflow.dataset_viewer alloy <substr>  # Query by alloy name")
        print("  python -m mgflow.dataset_viewer element <sym>   # Query by element")
        print("  python -m mgflow.dataset_viewer show <name>     # Show composition details")
        sys.exit(0)
    
    cmd = sys.argv[1]
    if cmd == "stats":
        show_stats()
    elif cmd == "all":
        sort = sys.argv[2] if len(sys.argv) > 2 else "name"
        show_all(sort_by=sort)
    elif cmd == "alloy" and len(sys.argv) > 2:
        query_alloy(sys.argv[2])
    elif cmd == "element" and len(sys.argv) > 2:
        query_element(sys.argv[2].capitalize())
    elif cmd == "show" and len(sys.argv) > 2:
        show_composition(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
