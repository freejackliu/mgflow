"""Inverse design loop: generate → extract → evaluate → select → repeat.

Integrates with mgflow's compute_all_features() and Predictor to serve
as the oracle/fitness function.
"""

import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    DEFAULT_ELEMENTS, LARGE_ELEMENT_POOL, DEFAULT_MODE,
    LATENT_DIM, GEN_HIDDEN_DIMS, GEN_DROPOUT, GEN_LEAKY_SLOPE,
    DEFAULT_N_GENERATIONS, DEFAULT_BATCH_SIZE, DEFAULT_LR, DEFAULT_BETA,
    DEFAULT_KEEP_TOP_K, DEFAULT_THRESHOLD, DEFAULT_SAVE_TOP_N,
    ALL_TARGETS,
)
from .objectives import scalarize
from .net import Generator
from .extract import extract_composition_batch, composition_to_dict, format_composition


def _compute_fitness(props: dict, target_cfg: dict) -> float:
    """Compute scalar fitness from predicted properties.

    Parameters
    ----------
    props : dict
        Prediction result from Predictor (Tg, Tx, Tl, E, H, BMG_prob, …).
    target_cfg : dict
        单目标：{"mode": "maximize"/"minimize"/"target", "key": ...}
        多目标：{"objectives": [{key, mode, weight}, ...], "normalize": "minmax"}

    Returns
    -------
    fitness : float
        Higher is better for ALL modes (minimize → negative).
    """
    # ── 多目标：加权求和标量化 ──
    if "objectives" in target_cfg:
        fit, _ = scalarize(
            props, target_cfg["objectives"],
            normalize=target_cfg.get("normalize", "minmax"),
        )
        return fit

    key = target_cfg["key"]

    # Handle composite keys
    if key == "Tx_minus_Tg":
        val = props.get("Tx", 0) - props.get("Tg", 0)
    elif key == "BMG_prob":
        val = props.get("BMG_prob", 0) or 0.0
    else:
        val = props.get(key, 0) or 0.0

    mode = target_cfg["mode"]
    if mode == "maximize":
        return float(val)
    elif mode == "minimize":
        return -float(val)
    elif mode == "target":
        target = target_cfg.get("target_value", 0)
        return -abs(float(val) - target)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _is_valid_composition(comp_dict: dict, min_elements: int = 2) -> bool:
    """Basic sanity check on a composition dict."""
    if len(comp_dict) < min_elements:
        return False
    total = sum(comp_dict.values())
    if not (99.5 <= total <= 100.5):
        return False
    return True


def run_inverse_design(
    target: str = "BMG",
    elements: list = None,
    mode: str = None,
    n_generations: int = DEFAULT_N_GENERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    keep_top_k: int = DEFAULT_KEEP_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    save_top_n: int = DEFAULT_SAVE_TOP_N,
    device: str = None,
    verbose: bool = True,
    output_dir: str = None,
):
    """Run the full inverse-design loop.

    Parameters
    ----------
    target : str or dict
        Predefined target name (see TARGETS in config.py) or a custom dict
        with keys "mode", "key", and optionally "target_value".
    elements : list of str, optional
        Candidate element symbols.
        In "fixed" mode: defaults to config.DEFAULT_ELEMENTS.
        In "auto"  mode: if not given, uses config.LARGE_ELEMENT_POOL (30+ elements);
                         if given, restricts the pool to the specified elements.
    mode : str, optional
        "fixed" — user-specified element set (default).
        "auto"  — large pool + generator auto-selects which elements to keep.
        Overrides config.DEFAULT_MODE.
    n_generations : int
        Number of generation iterations.
    batch_size : int
        Number of candidate compositions per generation.
    lr : float
        Learning rate for generator optimizer.
    keep_top_k : int
        Number of top candidates to retain per generation for reporting.
    threshold : float
        Minimum element fraction to keep (1% = 0.01).
    save_top_n : int
        Save top N candidates across all generations to JSON.
    device : str, optional
        "cuda", "cpu", or None for auto-detect.
    verbose : bool
        Print progress.
    output_dir : str, optional
        Directory for output files. Defaults to generator/output/.

    Returns
    -------
    result : dict
        {"candidates": [...], "best": {...}, "history": [...]}
    """
    # ── Parse target ──
    if isinstance(target, str):
        if target not in ALL_TARGETS:
            raise ValueError(f"Unknown target '{target}'. Options: {list(ALL_TARGETS.keys())}")
        target_cfg = dict(ALL_TARGETS[target])
    else:
        target_cfg = dict(target)

    # ── Mode & element set ──
    if mode is None:
        mode = DEFAULT_MODE

    if mode == "auto":
        pool = list(LARGE_ELEMENT_POOL)
        if elements is not None:
            # User-supplied elements: restrict pool to intersection
            elements = [el for el in elements if el in pool]
            if not elements:
                raise ValueError("No valid elements found in the large pool.")
        else:
            elements = pool
    else:  # "fixed"
        if elements is None:
            elements = list(DEFAULT_ELEMENTS)

    elements = list(elements)
    n_ele = len(elements)

    # ── Device ──
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ── Auto mode: limit max elements in composition to keep candidates realistic ──
    if mode == "auto":
        # For large pools, cap at ~8 elements per composition (matches GAN paper standard)
        if n_ele <= 10:
            max_ele_per_comp = n_ele  # small pool: allow all
        else:
            max_ele_per_comp = min(max(6, n_ele // 5), 10)  # 6~10 elements
        if verbose:
            print(f"  Auto mode: max {max_ele_per_comp} elements per composition "
                  f"(pool={n_ele} elements)")
    else:
        max_ele_per_comp = None

    # ── Output dir ──
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build generator ──
    gen = Generator(
        n_elements=n_ele,
        latent_dim=LATENT_DIM,
        hidden_dims=GEN_HIDDEN_DIMS,
        dropout=GEN_DROPOUT,
        leaky_slope=GEN_LEAKY_SLOPE,
    ).to(device)
    gen.train()

    optimizer = optim.Adam(gen.parameters(), lr=lr, betas=DEFAULT_BETA)

    # ── Late imports (mgflow oracle) ──
    from ..features.compute import compute_all_features
    from ..prediction.predictor import Predictor

    predictor = Predictor("pytorch")

    # ── Tracking ──
    all_candidates = []          # (fitness, generation, comp_dict, props)
    history = []                 # per-generation stats

    # ── Main loop ──
    start_time = time.time()

    for gen_idx in range(n_generations):
        # 1) Generate
        z = torch.empty(batch_size, LATENT_DIM, device=device).uniform_(-1, 1)
        with torch.no_grad():
            scores = gen(z).cpu().numpy()  # (batch, n_ele)

        # 2) Extract compositions
        comps = extract_composition_batch(
            scores, elements, threshold=threshold, max_elements=max_ele_per_comp,
        )

        # 3) Evaluate via mgflow oracle
        gen_fitnesses = []
        valid_comps = []
        for comp_dict in comps:
            if not _is_valid_composition(comp_dict):
                continue
            try:
                frac_dict = composition_to_dict(comp_dict)
                props = predictor.predict(frac_dict)
                fit = _compute_fitness(props, target_cfg)
            except Exception:
                continue
            gen_fitnesses.append((fit, gen_idx, comp_dict, props))
            valid_comps.append(comp_dict)

        # 4) Sort by fitness
        gen_fitnesses.sort(key=lambda x: x[0], reverse=True)

        # 5) Track top candidates
        all_candidates.extend(gen_fitnesses[:keep_top_k])

        # 6) Generator update (optional: reinforce good candidates)
        if len(gen_fitnesses) >= keep_top_k:
            top_scores = []
            for fit, _, comp_dict, _ in gen_fitnesses[:keep_top_k]:
                # Build target vector from top composition
                target_vec = np.zeros(n_ele, dtype=np.float32)
                for i, el in enumerate(elements):
                    target_vec[i] = comp_dict.get(el, 0.0) / 100.0
                if target_vec.sum() > 0:
                    target_vec = target_vec / target_vec.sum()
                top_scores.append(target_vec)

            if top_scores:
                target_tensor = torch.tensor(np.stack(top_scores), device=device)
                # Re-generate from noise and push toward top compositions
                z_train = torch.empty(keep_top_k, LATENT_DIM, device=device).uniform_(-1, 1)
                gen.train()
                optimizer.zero_grad()
                output = gen(z_train)
                loss = nn.MSELoss()(output, target_tensor)
                loss.backward()
                optimizer.step()
                gen_loss_val = loss.item()
        else:
            gen_loss_val = 0.0

        # 7) Record history
        if gen_fitnesses:
            best_fit = gen_fitnesses[0][0]
            mean_fit = np.mean([f[0] for f in gen_fitnesses]).item()
            n_valid = len(valid_comps)
        else:
            best_fit = -np.inf
            mean_fit = -np.inf
            n_valid = 0

        history.append({
            "generation": gen_idx,
            "best_fitness": best_fit,
            "mean_fitness": mean_fit,
            "n_valid": n_valid,
            "generator_loss": gen_loss_val,
        })

        # 8) Verbose progress
        if verbose and (gen_idx % 50 == 0 or gen_idx == n_generations - 1):
            elapsed = time.time() - start_time
            print(
                f"[{gen_idx:4d}/{n_generations}] "
                f"best={best_fit:+.4f}  mean={mean_fit:+.4f}  "
                f"valid={n_valid}/{batch_size}  "
                f"gen_loss={gen_loss_val:.6f}  "
                f"time={elapsed:.0f}s"
            )

            if gen_fitnesses:
                _, _, best_comp, best_props = gen_fitnesses[0]
                comp_str = format_composition(best_comp)
                print(f"  Best: {comp_str}")
                if best_props:
                    for pk in ["Tg", "Tx", "Tl", "E", "H"]:
                        pv = best_props.get(pk)
                        if pv is not None:
                            print(f"    {pk}: {pv:.1f}", end=" ")
                    bmg_p = best_props.get("BMG_prob")
                    if bmg_p is not None:
                        print(f"BMG_prob: {bmg_p:.3f}", end="")
                    print()

    # ── Finalise ──
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top_n = all_candidates[:save_top_n]

    # Serialize top results
    results_list = []
    for fit, gen_idx, comp_dict, props in top_n:
        results_list.append({
            "fitness": round(fit, 4),
            "generation": gen_idx,
            "composition": comp_dict,
            "composition_str": format_composition(comp_dict),
            "properties": {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in props.items()
                if v is not None
            } if props else {},
        })

    # Save to JSON
    output_path = output_dir / f"inverse_design_{target}.json"
    with open(output_path, "w") as f:
        json.dump({
            "target": target,
            "mode": mode,
            "elements": elements,
            "config": {
                "n_generations": n_generations,
                "batch_size": batch_size,
                "threshold": threshold,
            },
            "total_time_seconds": round(time.time() - start_time, 1),
            "top_candidates": results_list,
        }, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\nDone in {time.time() - start_time:.0f}s. Top {save_top_n} saved to {output_path}")

    return {
        "candidates": results_list,
        "best": results_list[0] if results_list else None,
        "history": history,
    }
