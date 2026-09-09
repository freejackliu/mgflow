"""
VAE-based composition refinement (微调模式).

Unlike the exploration-oriented Generator (net.py → loop.py), which freely
searches the full composition space with no training-data constraints,
the VAE refiner operates on the data manifold learned from 6453 known BMG
compositions.

Use case
--------
  1. Use run_inverse_design() or run_enhanced_inverse_design() for broad
     exploration → discover candidate systems (e.g. TaHfNbNiCo).
  2. Use refine_composition() to take those candidates and:
       • Generate nearby variants (fine-tune fractions)
       • Interpolate between two good candidates
       • Explore the local latent neighborhood

Key difference
--------------
  Generator (net.py)       → 任何元素组合都行，无数据约束 → 探索
  VAE (vae_refine.py)      → 只能在训练数据流形附近 → 微调

Usage
-----
  from mgflow.generator.vae_refine import refine_composition

  # Step 1: exploration (original generator)
  from mgflow.generator import run_inverse_design
  result = run_inverse_design(target='Tl_high', elements=['Ta','Hf','Nb','Ni','Co'])

  # Step 2: refinement around best candidate
  best_comp = result['best']['composition']
  refined = refine_composition(
      seed_compositions=[best_comp],
      n_variants=200,
      perturbation_scale=0.3,  # small = fine-tune, large = explore
      target='Tl_high',
  )
"""

import time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from .config import TARGETS
from .vae import (
    load_vae, CompositionVAE, CompositionDataset,
    VAE_SYMBOLS, N_ELEMENTS, ELEM_TO_VAE_IDX,
)
from ..data.loader import SYMBOL_TO_NO
from ..prediction.ensemble import EnsemblePredictor
from ..prediction.predictor import Predictor


# ===================================================================
# Composition ↔ VAE vector conversion
# ===================================================================

def composition_to_vae_vec(comp_dict, sum_to_one=True):
    """Convert {symbol: fraction} to VAE's 53-dim vector.

    Parameters
    ----------
    comp_dict : dict, {symbol: fraction_in_[0,1]_or_percent}
    sum_to_one : bool — renormalize after mapping

    Returns
    -------
    vec : ndarray (N_ELEMENTS,)
    """
    vec = np.zeros(N_ELEMENTS, dtype=np.float32)
    for sym, frac in comp_dict.items():
        # Accept both decimal (0~1) and percent (0~100)
        f = float(frac) / 100.0 if float(frac) > 1 else float(frac)
        atomic_no = SYMBOL_TO_NO.get(sym)
        if atomic_no is not None and atomic_no in ELEM_TO_VAE_IDX:
            vec[ELEM_TO_VAE_IDX[atomic_no]] = f
    if sum_to_one and vec.sum() > 0:
        vec /= vec.sum()
    return vec


def vae_vec_to_composition(vec, threshold=0.02):
    """Convert VAE's 53-dim vector to {symbol: percent} dict.

    Elements below threshold are dropped, survivors renormalized to 100%.
    """
    vec = np.asarray(vec, dtype=np.float64)
    idx = np.argsort(vec)[::-1]
    keep = idx[vec[idx] >= threshold]
    if len(keep) == 0:
        return {}
    fracs = vec[keep] / vec[keep].sum()
    return {VAE_SYMBOLS[j]: round(float(fracs[i]) * 100, 2)
            for i, j in enumerate(keep)}


# ===================================================================
# Refinement: generate variants near a seed composition
# ===================================================================

def refine_composition(
    seed_compositions,
    n_variants=200,
    perturbation_scale=0.3,
    target='Tl_high',
    elements=None,
    vae_latent_dim=32,
    vae_beta=0.05,
    oracle='ensemble',
    n_ensemble=30,
    threshold=0.025,
    min_elements=2,
    max_elements=8,
    verbose=True,
):
    """Generate refined variants near seed compositions using VAE latent space.

    Pipeline
    --------
      1. Encode each seed comp → latent mean μ
      2. Sample z = μ + ε·σ, where ε ~ N(0, perturbation_scale)
      3. Decode z → compositions
      4. Filter (threshold, element count)
      5. Evaluate with EnsemblePredictor
      6. Rank by target

    Parameters
    ----------
    seed_compositions : list of dict
        e.g. [{"Ta": 0.27, "Nb": 0.27, "Ni": 0.18, "Co": 0.17, "Hf": 0.11}]
    n_variants : int — number of variants to generate per seed
    perturbation_scale : float — noise std in latent space
        0.1 ~ very local fine-tune, 0.5 ~ broader exploration, 1.0 ~ far from seed
    target : str or dict — see TARGETS in config.py
    elements : list, optional — if set, filter to only these elements
    oracle : 'ensemble' or 'single' — which predictor to use
    n_ensemble : int — ensemble size
    threshold : float — minimum element fraction to keep
    min_elements, max_elements : int — filter by element count
    verbose : bool

    Returns
    -------
    dict with keys: "candidates", "best", "history", "seeds"
    """
    t0 = time.time()

    # ── Target ──
    if isinstance(target, str):
        if target not in TARGETS:
            raise ValueError(f"Unknown target '{target}'")
        target_cfg = dict(TARGETS[target])
    else:
        target_cfg = dict(target)

    key = target_cfg["key"]
    mode = target_cfg["mode"]

    # ── Load models ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(latent_dim=vae_latent_dim, beta=vae_beta, device=device)
    vae.eval()

    if verbose:
        print(f"VAE refiner loaded: {vae}")
        print(f"Seeds: {len(seed_compositions)} composition(s)")

    if oracle == 'ensemble':
        pred = EnsemblePredictor(n_models=n_ensemble)
    else:
        pred = Predictor("pytorch")

    # ── Encode seeds ──
    seed_vectors = []
    for comp in seed_compositions:
        vec = composition_to_vae_vec(comp)
        seed_vectors.append(vec)

    seed_zs = []
    with torch.no_grad():
        for vec in seed_vectors:
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
            mu, logvar = vae.encode(x)
            seed_zs.append(mu.cpu().numpy()[0])

    seed_zs = np.array(seed_zs)

    # ── Generate variants ──
    n_per_seed = max(1, n_variants // len(seed_zs))
    all_z = np.repeat(seed_zs, n_per_seed, axis=0)
    noise = np.random.randn(*all_z.shape) * perturbation_scale
    all_z += noise

    z_tensor = torch.tensor(all_z, dtype=torch.float32, device=device)
    with torch.no_grad():
        decoded = vae.decode(z_tensor).cpu().numpy()

    # ── Decode & evaluate ──
    all_candidates = []
    n_total = len(decoded)
    batch_size = 100

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_comps = []

        for c in decoded[start:end]:
            comp = vae_vec_to_composition(c, threshold=threshold)
            n_ele = len(comp)

            if n_ele < min_elements:
                continue
            if max_elements and n_ele > max_elements:
                continue
            if elements:
                comp = {k: v for k, v in comp.items() if k in elements}
                if len(comp) < min_elements:
                    continue

            batch_comps.append(comp)

        if not batch_comps:
            continue

        # Evaluate
        frac_batch = [{k: v/100.0 for k, v in c.items()} for c in batch_comps]
        try:
            results = pred.predict_batch(frac_batch)
        except Exception:
            continue

        for comp, props in zip(batch_comps, results):
            # Compute fitness
            if key == "Tx_minus_Tg":
                val = (props.get("Tx", 0) or 0) - (props.get("Tg", 0) or 0)
            elif key == "BMG_prob":
                val = props.get("BMG_prob", 0) or 0.0
            else:
                val = props.get(key, 0) or 0.0
            fit = float(val) if mode == "maximize" else -float(val)

            all_candidates.append((fit, comp, props))

    # ── Rank ──
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    n_valid = len(all_candidates)

    # ── Format results ──
    results_list = []
    for fit, comp, props in all_candidates[:200]:
        results_list.append({
            "fitness": round(fit, 4),
            "composition": comp,
            "composition_str": "".join(f"{s}{f:.1f}" for s,f in sorted(comp.items(), key=lambda x: -x[1])),
            "properties": {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in props.items()
                if v is not None and not k.endswith("_preds")
            },
        })

    history = {
        "n_seeds": len(seed_compositions),
        "n_variants_generated": n_total,
        "n_valid": n_valid,
        "perturbation_scale": perturbation_scale,
        "time_seconds": round(time.time() - t0, 1),
    }

    if verbose:
        elapsed = time.time() - t0
        print(f"\nRefinement complete: {n_valid}/{n_total} valid  [{elapsed:.0f}s]")
        if results_list:
            best = results_list[0]
            print(f"Best: {best['composition_str']}  "
                  f"fitness={best['fitness']:.4f}  "
                  f"{key}={best['properties'].get(key, '?'):.1f}")

    return {
        "candidates": results_list,
        "best": results_list[0] if results_list else None,
        "history": history,
        "seeds": seed_compositions,
    }


# ===================================================================
# Interpolation between two compositions
# ===================================================================

def interpolate_compositions(
    comp_a,
    comp_b,
    steps=10,
    target='Tl_high',
    verbose=True,
):
    """Interpolate between two compositions in VAE latent space.

    Parameters
    ----------
    comp_a, comp_b : dict — seed compositions
    steps : int — number of interpolation points
    target : str or dict

    Returns
    -------
    list of dict, each with composition, properties, and alpha (0→1)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_vae(latent_dim=32, beta=0.05, device=device)
    vae.eval()
    pred = EnsemblePredictor(n_models=30)

    v_a = torch.tensor(composition_to_vae_vec(comp_a), dtype=torch.float32).unsqueeze(0).to(device)
    v_b = torch.tensor(composition_to_vae_vec(comp_b), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        mu_a, _ = vae.encode(v_a)
        mu_b, _ = vae.encode(v_b)

    alphas = np.linspace(0, 1, steps)
    results = []

    for alpha in alphas:
        z = (1 - alpha) * mu_a + alpha * mu_b
        with torch.no_grad():
            decoded = vae.decode(z).cpu().numpy()[0]
        comp = vae_vec_to_composition(decoded, threshold=0.02)
        if not comp:
            continue

        frac = {k: v/100.0 for k, v in comp.items()}
        try:
            props = pred.predict(frac)
        except Exception:
            continue

        results.append({
            "alpha": round(float(alpha), 3),
            "composition": comp,
            "composition_str": "".join(f"{s}{f:.1f}" for s,f in sorted(comp.items(), key=lambda x: -x[1])),
            "properties": props,
        })

    if verbose:
        print(f"Interpolation {steps} steps:")
        for r in results:
            tl = r['properties'].get('Tl', 0) or 0
            print(f"  α={r['alpha']:.2f}  {r['composition_str']:40s}  Tl={tl:.0f}")

    return results
